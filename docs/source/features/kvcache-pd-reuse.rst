.. _kvcache-pd-reuse:

============================================
KV-Centric P/D: Offloading vs PD Reuse
============================================

AIBrix exposes two engine-side vLLM KV connectors that look similar but solve different
problems. This page explains **when to use which**, what the **shared store** is, how to
pick an **L2 backend**, and how to run prefill/decode disaggregation on a **single node
without RDMA**.

.. note::
    This page is about the **engine-side** vLLM ``kv_connector`` (set in
    ``--kv-transfer-config``). It builds on the L1/L2 concepts in
    :ref:`kvcache-offloading` and the framework design in
    :ref:`aibrix_kvcache-offloading-framework`. For **gateway-side** prefill/decode
    *routing* (the ``pd`` routing strategy, pod labels, prompt-length bucketing), see
    :ref:`pd_disaggregation`.

.. contents:: On this page
   :local:
   :depth: 2


The one story: one cache, two jobs
----------------------------------

The same tiered KV cache that offloads attention key/value tensors out of GPU memory can
also hand that KV from a **prefill** engine to a **decode** engine. AIBrix splits this into
two connectors so each can be tuned independently:

- **AIBrixOffloadingConnector** — offloads KV out of scarce GPU HBM into a tiered cache
  (L1 DRAM + optional L2 remote store). Goal: **more effective context capacity** and
  **cross-engine prefix reuse**. This is the connector demonstrated in
  :ref:`kvcache-offloading`.
- **AIBrixPDReuseConnector** — combines **prefill→decode disaggregation** with **KV
  reuse**. The prefill engine produces the prompt KV; the decode engine consumes it through
  a **shared L2 store** instead of recomputing the prompt. The L2 backend (e.g. SHFS) is
  transparent to the connector.

.. note::
    Two layers share the word "connector". The **engine-side** ``kv_connector`` on this page
    (``AIBrixOffloadingConnector`` / ``AIBrixPDReuseConnector``) is *inside* the inference
    engine and moves KV bytes. The **gateway-side** KV-transfer agent
    (``AIBRIX_KV_CONNECTOR_TYPE`` = ``shfs`` / ``nixl`` / ``mooncake``, see
    :ref:`pd_disaggregation`) lives in the router and only tells the decode pod *where* to
    pull KV from. They are different layers.


Two connectors at a glance
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Aspect
     - ``AIBrixOffloadingConnector``
     - ``AIBrixPDReuseConnector``
   * - Primary goal
     - Offload KV from HBM → capacity + cross-engine prefix reuse
     - Prefill→decode KV handoff **and** reuse
   * - Topology
     - Single engine (or many engines sharing an L2)
     - Disaggregated prefill + decode roles
   * - KV path
     - HBM → L1 (DRAM) → optional L2 (remote)
     - Prefill writes L2 (shared store) → decode reads L2
   * - vLLM ``kv_role``
     - n/a (standalone offloading)
     - ``kv_both`` on prefill, ``kv_consumer`` on decode
   * - Typical L2 backend
     - InfiniStore / PriSKV / HPKV (capacity tiers)
     - SHFS shared store (also PriSKV / InfiniStore)
   * - RDMA required?
     - No for L1-only; backend-dependent for L2
     - **No** with SHFS — the single-node, no-RDMA path

Both are selected the same way — via the vLLM ``--kv-transfer-config`` flag:

.. code-block:: text

    # Offloading (capacity + reuse), standalone engine
    --kv-transfer-config '{"kv_connector":"AIBrixOffloadingConnector","kv_role":"kv_both"}'

    # PD reuse: prefill engine
    --kv-transfer-config '{"kv_connector":"AIBrixPDReuseConnector","kv_role":"kv_both","engine_id":"prefill_0"}'

    # PD reuse: decode engine
    --kv-transfer-config '{"kv_connector":"AIBrixPDReuseConnector","kv_role":"kv_consumer","engine_id":"decode_0"}'


Comparison: three KV-transfer paths
------------------------------------

The figure below contrasts the three ways KV moves between engines in AIBrix.

.. mermaid::

    flowchart TB
        subgraph A["(a) AIBrixOffloadingConnector"]
            direction TB
            ae["vLLM engine"] -->|offload| al1["L1: DRAM"]
            al1 -->|spill / ingest| al2["L2: remote store<br/>InfiniStore / PriSKV"]
            al2 -. cross-engine reuse .-> ae2["other vLLM engines"]
        end
        subgraph B["(b) PD via NixlConnector"]
            direction TB
            bp["Prefill engine"] -->|point-to-point<br/>RDMA / network| bd["Decode engine"]
        end
        subgraph C["(c) AIBrixPDReuseConnector"]
            direction TB
            cp["Prefill engine<br/>kv_role: kv_both"] -->|write KV| cs["Shared L2 store<br/>(SHFS / PriSKV)"]
            cs -->|read KV| cd["Decode engine<br/>kv_role: kv_consumer"]
        end

Path **(a)** is pure offloading — capacity and reuse, no disaggregation. Path **(b)** moves
KV directly from prefill to decode over a network/RDMA transport (``NixlConnector``); it has
no shared store. Path **(c)**, the focus of this page, routes the handoff *through a shared
store*, which is why it works without RDMA and naturally supports reuse.


How PD reuse works
------------------

In PD reuse, prefill and decode are separate engines that meet at the shared store. The
prefill engine runs ``kv_role: kv_both`` (it both produces KV and can reuse cached KV); the
decode engine runs ``kv_role: kv_consumer``. Each engine carries a unique ``engine_id``.

.. mermaid::

    sequenceDiagram
        participant Cl as Client
        participant Pf as Prefill (kv_both)
        participant St as Shared store (SHFS L2)
        participant Dc as Decode (kv_consumer)

        Cl->>Pf: prompt
        Pf->>Pf: compute prompt KV
        Pf->>St: write KV blocks
        Pf-->>Cl: first token
        Dc->>St: read KV blocks (skip recompute)
        Dc-->>Cl: stream remaining tokens

Because the KV lands in a shared store, a later request that reuses the same prefix can hit
the cache directly — disaggregation and prefix reuse fall out of the same mechanism.


The shared store (SHFS)
-----------------------

The **shared store** is the L2 backend that both prefill and decode engines read and write.
The simplest backend is **SHFS** (Shared File System): the ``SHFSConnector`` writes each KV
block as a file under a shared directory that every engine mounts at ``AIBRIX_KV_CACHE_OL_SHFS_ROOT``.

.. code-block:: yaml

    env:
      - name: AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED   # KV is shared via L2 only
        value: "0"
      - name: AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND
        value: "SHFS"
      - name: AIBRIX_KV_CACHE_OL_SHFS_ROOT
        value: "/kvcache"                            # same mount on prefill + decode

Key properties:

- **No RDMA, no special NICs.** SHFS is plain file I/O, so it runs anywhere a directory can
  be shared.
- **Same directory on every engine.** Co-located on one node? Use a node-local ``hostPath``.
  Spread across nodes? Back ``/kvcache`` with a shared volume (NFS or an RWX CSI PVC).
- **Reuse for free.** Blocks written by one engine are visible to all — cross-engine prefix
  reuse without a separate cache cluster.


Backend choices
---------------

The L2 backend is selected with ``AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND``. Pick by your
topology and fabric:

.. list-table::
   :header-rows: 1
   :widths: 16 22 26 36

   * - Backend
     - Transport
     - Best for
     - Notes
   * - ``SHFS``
     - Shared file system (no RDMA)
     - Single node, or any nodes sharing a directory
     - Simplest PD-reuse path. hostPath (1 node) or NFS / RWX PVC (multi-node).
   * - ``PRISKV``
     - TCP (Redis protocol, port 6379)
     - No-RDMA multi-node, moderate scale
     - Works over plain TCP; optional ``MPUT/MGET`` and zero-copy.
   * - ``INFINISTORE``
     - RDMA (or TCP)
     - High-throughput, RDMA-capable clusters
     - Lowest latency at scale; set ``CONNECTION_TYPE=RDMA`` and a device list. TCP mode exists for testing.
   * - ``HPKV``
     - TCP / RDMA
     - High-performance shared KV store
     - Alternative high-performance backend.

.. note::
    **Single-node, no-RDMA path:** use **SHFS** (a node-local shared directory) — or
    **PriSKV over TCP** if you prefer a service. Neither needs an RDMA NIC, RoCE, or
    InfiniBand. Reach for **InfiniStore** (RDMA) only when you have the fabric and need the
    throughput.

Full per-backend environment-variable tables live in the framework reference:
:ref:`aibrix_kvcache-offloading-framework` → *Environment Variables Reference*.


Configure PD reuse end to end
------------------------------

The sample below disaggregates a model into a prefill role (``kv_both``) and a decode role
(``kv_consumer``), both sharing an SHFS store mounted at ``/kvcache``. It is a trimmed,
runnable version of the regression fixture in
``python/aibrix_kvcache/tests/pd_reuse/vllm-pd-reuse.yaml``.

.. literalinclude:: ../../../samples/kvcache/pd-reuse/vllm-pd-reuse.yaml
   :language: yaml
   :linenos:

.. code-block:: console

    $ kubectl apply -f samples/kvcache/pd-reuse/vllm-pd-reuse.yaml

    stormservice.orchestration.aibrix.ai/qwen3-32b-pd-reuse created

.. note::
    * The ``engine_id`` must be unique per engine (the sample derives it from
      ``ROLE_REPLICA_INDEX``).
    * ``role-name``/``roleset-name`` labels let the gateway pair prefill and decode pods; see
      :ref:`pd_disaggregation` for enabling the ``pd`` routing strategy.
    * For multi-node, replace the ``kvcache`` ``hostPath`` volume with a shared volume (NFS or
      an RWX PVC) so both roles see the same files.


Choosing your path
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Your situation
     - Recommendation
   * - Single node, no RDMA, want P/D
     - ``AIBrixPDReuseConnector`` + **SHFS** shared store
   * - Multi-node, no RDMA, want P/D
     - ``AIBrixPDReuseConnector`` + **PriSKV** (TCP), or SHFS on a shared volume
   * - RDMA fabric, throughput-critical P/D
     - ``AIBrixPDReuseConnector`` + **InfiniStore**, or PD via ``NixlConnector``
   * - Just need capacity / cross-engine prefix reuse (no P/D split)
     - ``AIBrixOffloadingConnector`` (L1 DRAM, optional L2)


Related
-------

- :ref:`kvcache-offloading` — L1/L2 offloading setup with the AIBrixOffloadingConnector.
- :ref:`aibrix_kvcache-offloading-framework` — framework design + full env-var reference.
- :ref:`pd_disaggregation` — gateway-side prefill/decode routing and pod labeling.
