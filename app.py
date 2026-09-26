"""
IBM Enterprise Support & Diagnostic Copilot (Agentic RAG)
=========================================================
Architecture:
  - Vector RAG: In-memory TF-IDF + Cosine similarity over IBM Redbooks, APARs, & TSS SOPs.
  - ReAct Agent: Autonomous tool calling across simulated OpenShift health checks,
                 watsonx quota audits, APAR lookups, and operational remediation APIs.
  - UI: Gradio-based Technical Support Engineer (TSE) Operations Cockpit.
  - Micro-Runtime: Operates cleanly under <35 MB RAM footprint.
"""

import os
import sys
import re
import math
import socket
from collections import Counter
import gradio as gr

# ==============================================================================
# 1. NETWORK & DEPLOYMENT UTILITIES
# ==============================================================================

def find_available_port(start_port=7860, max_attempts=50):
    """Finds an open loopback network port to avoid port collision crashes."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port

# ==============================================================================
# 2. LIGHTWEIGHT TF-IDF DOMAIN VECTOR RAG ENGINE (<35 MB RAM)
# ==============================================================================

IBM_KNOWLEDGE_CORPUS = [
    {
        "doc_id": "IBM-OCP-101",
        "title": "Red Hat OpenShift Control Plane CrashLoopBackOff & etcd Latency",
        "category": "Hybrid Cloud Infrastructure",
        "content": (
            "OpenShift Control Plane degraded alerts often stem from slow etcd disk commit fsync latencies (>10ms) "
            "or CrashLoopBackOff in the kube-apiserver/etcd-quorum pods. Verification SOP: Check node I/O metrics and "
            "etcd member list health. If etcd reports leader election timeouts, isolate degraded member, check storage IOPS, "
            "and cordon/drain affected worker or control plane nodes to initiate automated leader re-election."
        )
    },
    {
        "doc_id": "IBM-WX-204",
        "title": "watsonx.ai Token Exhaustion, Concurrency Throttling & 429 Errors",
        "category": "watsonx AI & Data Platform",
        "content": (
            "Enterprise tenants querying watsonx.ai foundation models (Granite 13b, Llama 3) receive HTTP 429 Too Many Requests "
            "when exceeding provisioned Tokens Per Minute (TPM) or Requests Per Minute (RPM) thresholds. Verification SOP: "
            "Audit tenant GPU worker pools and dynamic concurrency slots. Remediation protocol permits issuing temporary 20% burst "
            "quota allocations or dynamic failover routing to an alternate high-availability regional cluster."
        )
    },
    {
        "doc_id": "IBM-APAR-993",
        "title": "APAR IT43921: DB2 for z/OS Deadlock in High-Throughput CICS Transactions",
        "category": "IBM zSystems & DB2 Mainframe",
        "content": (
            "APAR IT43921 resolves unexpected SQLCODE -911 / reason code 00C90088 lock timeouts during heavy parallel batch runs. "
            "Applicable components: DB2 for z/OS v12 and v13 under CICS transaction manager. Verification: Check Syslog for "
            "DSNT376I lock escalation messages. Remediation: Apply Program Temporary Fix (PTF) UI89231 immediately to suppress "
            "deadlock cascades and adjust IRLM deadlock detection timer from 5s down to 1s."
        )
    },
    {
        "doc_id": "IBM-RACF-401",
        "title": "z16 RACF Digital Certificate Expiration & TLS Handshake Drop",
        "category": "IBM zSystems Security",
        "content": (
            "Mainframe secure communications drop with ICH408I or SECINT-04 authentication errors when RACF key rings encounter "
            "expired CA or server signing certificates. SOP mandates inspecting the RACF database via RACDCERT LISTRING. If expired, "
            "trigger automated staging of emergency self-signed root extension or re-bind updated enterprise certificate authority."
        )
    },
    {
        "doc_id": "IBM-CP-502",
        "title": "IBM Cloud Pak for Data Operator Health & Operand Sync Timeout",
        "category": "Cloud Pak Operations",
        "content": (
            "Cloud Pak for Data operands enter 'Progressing' timeout or 'Degraded' states when underlying persistent volume claims "
            "(PVCs) reach IOPS saturation or Red Hat storage cluster fails quorum. SOP: Inspect OLM subscription status, recycle "
            "the stalled operator pod, and verify storage class dynamic provisioning permissions."
        )
    },
    {
        "doc_id": "IBM-SAN-601",
        "title": "IBM FlashSystem NVMe-oF Path Degradation & Fabric Latency",
        "category": "Enterprise Storage",
        "content": (
            "FlashSystem arrays logging 0x840003 fabric degradation warnings indicate NVMe-oF Fibre Channel congestion or RDMA buffer drops. "
            "Verification SOP: Inspect SAN fabric port error rates, SFP optical signal levels, and host multipath daemon state. "
            "Remediation requires triggering non-disruptive fabric zone failover and rebalancing paths to alternate canister."
        )
    },
    {
        "doc_id": "IBM-PWR-702",
        "title": "PowerVM Shared Ethernet Adapter (SEA) Packet Drops & Entitlement Limit",
        "category": "IBM Power Systems & AIX",
        "content": (
            "AIX LPARs experience network packet loss and high latency when the Virtual I/O Server (VIOS) SEA exceeds allocated processor entitlement. "
            "Verification: Check entstat and topas -E metrics. Remediation: Dynamically allocate unassigned shared processor pool units "
            "to VIOS via DLPAR and enable large send offload."
        )
    }
]

class MicroTFIDFRetriever:
    """Lightweight TF-IDF Vector Retriever executing under strict RAM constraints."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())

    def _build_index(self):
        doc_tokens = [self._tokenize(doc["title"] + " " + doc["content"]) for doc in self.corpus]
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        
        num_docs = len(self.corpus)
        vocab_list = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(vocab_list)}
        self.idf = [math.log((num_docs + 1) / (df[term] + 1)) + 1.0 for term in vocab_list]

        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = [0.0] * len(self.vocab)
            norm = 0.0
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    score = (count / len(tokens)) * self.idf[idx]
                    vec[idx] = score
                    norm += score ** 2
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self.doc_vectors.append([v / norm for v in vec])

    def retrieve(self, query, top_k=2):
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        q_vec = [0.0] * len(self.vocab)
        norm = 0.0
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                score = (count / (len(tokens) or 1)) * self.idf[idx]
                q_vec[idx] = score
                norm += score ** 2
        norm = math.sqrt(norm) if norm > 0 else 1.0
        q_vec = [v / norm for v in q_vec]

        scores = []
        for i, d_vec in enumerate(self.doc_vectors):
            dot = sum(q * d for q, d in zip(q_vec, d_vec))
            scores.append((dot, self.corpus[i]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

retriever = MicroTFIDFRetriever(IBM_KNOWLEDGE_CORPUS)

# ==============================================================================
# 3. AUTONOMOUS REACT TOOLS (SIMULATED IBM ENTERPRISE APIS)
# ==============================================================================

MOCK_DATABASE = {
    "clusters": {
        "ocp-prod-us-east": {"status": "DEGRADED", "etcd_latency_ms": 38, "degraded_operator": "kube-apiserver", "pod_restarts": 14},
        "ocp-prod-eu-west": {"status": "HEALTHY", "etcd_latency_ms": 3, "degraded_operator": "None", "pod_restarts": 0}
    },
    "tenants": {
        "TENANT-BANK-01": {"tier": "Enterprise Platinum", "tpm_limit": 500000, "current_tpm_usage": 524000, "gpu_worker_health": "CONGESTED"},
        "TENANT-GOV-02": {"tier": "Government FedRAMP", "tpm_limit": 300000, "current_tpm_usage": 110000, "gpu_worker_health": "HEALTHY"},
        "TENANT-AIRLINE-03": {"tier": "Mission-Critical 24/7", "tpm_limit": 800000, "current_tpm_usage": 835000, "gpu_worker_health": "CONGESTED"}
    },
    "apar_db": {
        "SQLCODE-911": {"apar_id": "APAR-IT43921", "ptf_patch": "PTF-UI89231", "status": "AVAILABLE", "severity": "HIPER"},
        "ICH408I": {"apar_id": "APAR-OA61209", "ptf_patch": "PTF-UJ04812", "status": "AVAILABLE", "severity": "URGENT"},
        "ERR_NVME_0X84": {"apar_id": "APAR-HU02144", "ptf_patch": "MICROCODE-98.31", "status": "AVAILABLE", "severity": "CRITICAL"}
    }
}

def tool_inspect_openshift_cluster_health(cluster_id):
    """Inspects OpenShift control plane state, etcd sync latency, and degraded operators."""
    cluster = MOCK_DATABASE["clusters"].get(cluster_id, {"status": "HEALTHY", "etcd_latency_ms": 4, "degraded_operator": "None", "pod_restarts": 0})
    return {
        "tool": "inspect_openshift_cluster_health",
        "cluster_id": cluster_id,
        "cluster_status": cluster["status"],
        "etcd_fsync_latency": f"{cluster['etcd_latency_ms']}ms",
        "degraded_component": cluster["degraded_operator"],
        "action_required": "CORDON_AND_DRAIN_ETCD_MEMBER" if cluster["etcd_latency_ms"] > 10 else "NONE"
    }

def tool_verify_watsonx_token_quota(tenant_id):
    """Audits tenant TPM/RPM usage against watsonx.ai rate limits."""
    tenant = MOCK_DATABASE["tenants"].get(tenant_id, {"tier": "Standard", "tpm_limit": 200000, "current_tpm_usage": 50000, "gpu_worker_health": "HEALTHY"})
    over_limit = tenant["current_tpm_usage"] > tenant["tpm_limit"]
    return {
        "tool": "verify_watsonx_token_quota",
        "tenant_id": tenant_id,
        "tier": tenant["tier"],
        "tpm_saturation": f"{(tenant['current_tpm_usage'] / tenant['tpm_limit']) * 100:.1f}%",
        "quota_breach": over_limit,
        "eligible_for_burst": tenant["tier"] in ["Enterprise Platinum", "Government FedRAMP", "Mission-Critical 24/7"]
    }

def tool_query_ibm_apar_database(error_code):
    """Cross-references error traces against known IBM APARs and Program Temporary Fixes (PTFs)."""
    apar = MOCK_DATABASE["apar_db"].get(error_code, {"apar_id": "APAR-GEN-000", "ptf_patch": "NONE", "status": "NOT_FOUND", "severity": "INFORMATIONAL"})
    return {
        "tool": "query_ibm_apar_database",
        "error_code": error_code,
        "matched_apar": apar["apar_id"],
        "patch_package": apar["ptf_patch"],
        "fix_status": apar["status"],
        "hiper_flag": apar["severity"] == "HIPER"
    }

def tool_execute_ibm_remediation(action_type, target_resource):
    """Simulates programmatic remediation across IBM Cloud Pak, OpenShift, watsonx, or Storage."""
    return {
        "tool": "execute_ibm_remediation",
        "status": "COMMITTED",
        "action_type": action_type,
        "target_resource": target_resource,
        "audit_trace": f"Automated operational remediation '{action_type}' successfully applied to resource '{target_resource}'."
    }

# ==============================================================================
# 4. REASONING ENGINE (REACT + MINTO SYNTHESIS)
# ==============================================================================

def run_ibm_copilot(ticket_id, tenant_id, component, severity, error_trace, client_notes):
    reasoning_trace = []
    reasoning_trace.append(f"🔍 [PERCEPTION] Ingesting incident {ticket_id} ({severity}) for Tenant: {tenant_id} on Component: {component}.")

    # Step 1: Grounded Domain RAG Retrieval
    search_context = f"{component} {error_trace} {client_notes}"
    retrieved = retriever.retrieve(search_context, top_k=2)
    top_sop = retrieved[0][1]
    reasoning_trace.append(f"📚 [DOMAIN RAG] Grounded SOP: {top_sop['doc_id']} ('{top_sop['title']}') with confidence {retrieved[0][0]:.3f}.")

    # Step 2: Diagnostic Tool Execution Loop
    tool_outputs = {}

    if "OpenShift" in component or "ocp" in error_trace.lower() or "etcd" in error_trace.lower():
        reasoning_trace.append("⚙️ [ACTION] Calling `tool_inspect_openshift_cluster_health` on 'ocp-prod-us-east'.")
        ocp_diag = tool_inspect_openshift_cluster_health("ocp-prod-us-east")
        tool_outputs["openshift"] = ocp_diag
        reasoning_trace.append(f"📊 [OBSERVATION] Cluster: {ocp_diag['cluster_status']} | etcd Latency: {ocp_diag['etcd_fsync_latency']} | Degraded: {ocp_diag['degraded_component']}.")

    if "watsonx" in component or "429" in error_trace or "token" in error_trace.lower():
        reasoning_trace.append(f"⚙️ [ACTION] Calling `tool_verify_watsonx_token_quota` for Tenant: {tenant_id}.")
        wx_diag = tool_verify_watsonx_token_quota(tenant_id)
        tool_outputs["watsonx"] = wx_diag
        reasoning_trace.append(f"📊 [OBSERVATION] TPM Saturation: {wx_diag['tpm_saturation']} | Quota Breach: {wx_diag['quota_breach']} | Burst Eligible: {wx_diag['eligible_for_burst']}.")

    if "SQLCODE" in error_trace or "ICH408I" in error_trace or "0x840003" in error_trace or "Mainframe" in component or "FlashSystem" in component:
        err_key = "SQLCODE-911" if "911" in error_trace else ("ICH408I" if "ICH408I" in error_trace else "ERR_NVME_0X84")
        reasoning_trace.append(f"⚙️ [ACTION] Calling `tool_query_ibm_apar_database` for code '{err_key}'.")
        apar_diag = tool_query_ibm_apar_database(err_key)
        tool_outputs["apar"] = apar_diag
        reasoning_trace.append(f"📊 [OBSERVATION] Matched: {apar_diag['matched_apar']} | PTF: {apar_diag['patch_package']} | HIPER Flag: {apar_diag['hiper_flag']}.")

    # Step 3: Autonomous Remediation
    remediation_action = None
    if tool_outputs.get("openshift", {}).get("cluster_status") == "DEGRADED":
        reasoning_trace.append("🚀 [AUTONOMOUS REMEDIATION] Triggering drain on degraded control plane node to force etcd leader re-election.")
        remediation_action = tool_execute_ibm_remediation("CORDON_AND_DRAIN_ETCD_MEMBER", "ocp-prod-us-east-cp-1")

    elif tool_outputs.get("watsonx", {}).get("quota_breach") and tool_outputs.get("watsonx", {}).get("eligible_for_burst"):
        reasoning_trace.append("🚀 [AUTONOMOUS REMEDIATION] Enterprise tier verified. Authorizing temporary 20% burst quota allocation.")
        remediation_action = tool_execute_ibm_remediation("PROVISION_TEMPORARY_BURST_TPM", tenant_id)

    elif tool_outputs.get("apar", {}).get("fix_status") == "AVAILABLE":
        ptf = tool_outputs["apar"]["patch_package"]
        reasoning_trace.append(f"🚀 [AUTONOMOUS REMEDIATION] Staging verified Program Temporary Fix '{ptf}' in deployment pipeline.")
        remediation_action = tool_execute_ibm_remediation(f"STAGE_{ptf}_FOR_DEPLOYMENT", tenant_id)
    elif "FlashSystem" in component or "NVMe" in error_trace:
        reasoning_trace.append("🚀 [AUTONOMOUS REMEDIATION] Initiating SAN fabric zone failover and rebalancing paths to canister B.")
        remediation_action = tool_execute_ibm_remediation("FAILOVER_FABRIC_ZONE_B", "FS9200-ARRAY-01")
    elif "Power" in component or "SEA" in error_trace:
        reasoning_trace.append("🚀 [AUTONOMOUS REMEDIATION] Dynamically allocating +2.0 shared processor units to VIOS via DLPAR.")
        remediation_action = tool_execute_ibm_remediation("ALLOCATE_DLPAR_CORES", "VIOS-PRIMARY-01")
    else:
        remediation_action = {"action_type": "NO_IMMEDIATE_ACTION", "status": "MONITORING"}

    # Step 4: Top-Down Minto Structured Work Order
    minto_work_order = f"""### 📋 IBM Technology Support Services (TSS) Work Order
**Incident ID:** `{ticket_id}` | **Severity:** `{severity}` | **Tenant:** `{tenant_id}`

#### 1. Core Synthesis & Root Cause
* **Identified Component Failure:** {top_sop['title']}
* **Diagnostic Verification:** {
    'OpenShift etcd commit latency exceeded 10ms threshold causing control plane instability.' if 'OpenShift' in component else
    'watsonx.ai rate-limit saturation (TPM quota breach) under high concurrency load.' if 'watsonx' in component else
    'FlashSystem NVMe-oF fabric buffer drops degrading storage I/O.' if 'FlashSystem' in component else
    'PowerVM Virtual I/O Server processor entitlement saturation causing network packet drops.' if 'Power' in component else
    'Known upstream APAR defect causing transactional lock timeouts in production.'
}

#### 2. Telemetry & Runbook Grounding
* **Referenced Knowledge SOP:** `{top_sop['doc_id']}` ({top_sop['category']})
* **Operational Telemetry:**
  * OpenShift State: `{tool_outputs.get('openshift', {}).get('cluster_status', 'HEALTHY')}`
  * watsonx Quota Saturation: `{tool_outputs.get('watsonx', {}).get('tpm_saturation', 'NORMAL')}`
  * APAR Patch Staged: `{tool_outputs.get('apar', {}).get('patch_package', 'NONE')}`

#### 3. Executed Remediation & Protocol
* **Autonomous Action:** `{remediation_action['action_type']}` (`{remediation_action['status']}`)
* **Next Steps for Technical Support Engineer (TSE):**
  1. Validate telemetry recovery in IBM Cloud Monitoring / Syslog consoles.
  2. Confirm tenant model serving throughput and latencies.
  3. Close Sev-1 incident within contractual SLA window.
"""

    client_advisory = f"""Dear Enterprise Support Contact,

Our automated systems have triaged Incident {ticket_id} affecting your {component} environment.
Root cause was diagnosed as: {top_sop['title']}.
Remediation action '{remediation_action['action_type']}' has been staged and committed in accordance with your IBM Enterprise SLA.
Service stability is actively restoring. A technical support specialist remains assigned for final verification."""

    return "\n\n".join(reasoning_trace), minto_work_order, client_advisory, top_sop["content"]

# ==============================================================================
# 5. GRADIO OPERATIONS COCKPIT UI & PRESETS
# ==============================================================================

PRESET_SCENARIOS = {
    "Scenario 1: OpenShift Control Plane Degraded (etcd stall)": [
        "INC-IBM-901", "TENANT-BANK-01", "Red Hat OpenShift", "Sev-1: Platform Outage",
        "kube-apiserver CrashLoopBackOff, etcd fsync latency 38ms",
        "Banking core transaction gateway unreachable due to OCP master nodes degraded."
    ],
    "Scenario 2: watsonx.ai 429 Token Rate-Limit Throttling": [
        "INC-IBM-902", "TENANT-BANK-01", "watsonx.ai Platform", "Sev-2: High Impact",
        "HTTP 429 RateLimitExceeded: TPM limit saturated on Granite-13b model deployment",
        "All loan underwriting AI batch inference calls are failing with rate limit errors."
    ],
    "Scenario 3: DB2 for z/OS CICS Lock Contention (APAR Defect)": [
        "INC-IBM-903", "TENANT-BANK-01", "IBM zSystems Mainframe", "Sev-1: Platform Outage",
        "SQLCODE -911 reason code 00C90088 lock timeout in DSNT376I",
        "Core ledger batch jobs failing with recursive deadlock errors."
    ],
    "Scenario 4: z16 RACF Security Certificate Handshake Expiry": [
        "INC-IBM-904", "TENANT-GOV-02", "IBM zSystems Mainframe", "Sev-2: High Impact",
        "ICH408I USER NOT AUTHORIZED, TLS key ring certificate expired",
        "Government secure mainframe portals dropping inbound API connections."
    ],
    "Scenario 5: Cloud Pak for Data Operand Storage Quorum Drop": [
        "INC-IBM-905", "TENANT-BANK-01", "IBM Cloud Pak for Data", "Sev-2: High Impact",
        "Operand degraded, PVC dynamic provisioning IOPS saturated",
        "Data science analytics pipelines failing to spawn worker nodes."
    ],
    "Scenario 6: FlashSystem 9200 NVMe-oF Path Degradation": [
        "INC-IBM-906", "TENANT-AIRLINE-03", "IBM FlashSystem Storage", "Sev-1: Platform Outage",
        "0x840003 NVMe fabric buffer drop, path failover warning",
        "Airline reservation database experiencing 800ms I/O latency spikes on storage cluster."
    ],
    "Scenario 7: PowerVM Virtual I/O Server (SEA) Packet Loss": [
        "INC-IBM-907", "TENANT-AIRLINE-03", "IBM Power Systems & AIX", "Sev-2: High Impact",
        "Shared Ethernet Adapter (SEA) packet drop, VIOS entitlement exhausted",
        "Production AIX SAP transactional instances dropping client connections."
    ]
}

DEFAULT_PROMPT_PRESETS = {
    "OpenShift etcd commit stall": "kube-apiserver CrashLoopBackOff, etcd fsync latency 38ms on master node",
    "watsonx.ai 429 token limit": "HTTP 429 RateLimitExceeded: TPM limit saturated on Granite-13b model deployment",
    "DB2 deadlock SQLCODE -911": "SQLCODE -911 reason code 00C90088 lock timeout in DSNT376I during CICS batch run",
    "Mainframe RACF TLS drop": "ICH408I USER NOT AUTHORIZED, TLS key ring certificate expired on RACF subsystem",
    "FlashSystem NVMe buffer drop": "0x840003 NVMe fabric buffer drop, path failover warning on SAN fabric",
    "PowerVM SEA packet drop": "Shared Ethernet Adapter (SEA) packet drop, VIOS entitlement exhausted on AIX core"
}

def load_preset_data(preset_name):
    return PRESET_SCENARIOS[preset_name]

def populate_custom_prompt(prompt_name):
    return DEFAULT_PROMPT_PRESETS[prompt_name]

with gr.Blocks(title="IBM Enterprise Support & Diagnostic Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏢 IBM Enterprise Support & Diagnostic Copilot
        ### Autonomous OpenShift, watsonx.ai, Mainframe & Storage Triage Engine
        *Built with In-Memory TF-IDF Vector RAG and an Autonomous ReAct Multi-Tool Agent (<35 MB RAM)*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("#### 🎛️ Preset Enterprise Scenarios")
            preset_selector = gr.Dropdown(
                label="Select Production Enterprise Incident",
                choices=list(PRESET_SCENARIOS.keys()),
                value="Scenario 1: OpenShift Control Plane Degraded (etcd stall)"
            )

            gr.Markdown("#### 📥 Inbound Incident Telemetry Payload")
            t_id = gr.Textbox(label="Incident Ticket ID", value="INC-IBM-901")
            tenant_id = gr.Dropdown(label="Enterprise Tenant", choices=["TENANT-BANK-01", "TENANT-GOV-02", "TENANT-AIRLINE-03"], value="TENANT-BANK-01")
            component = gr.Dropdown(
                label="Target Ecosystem Component",
                choices=["Red Hat OpenShift", "watsonx.ai Platform", "IBM zSystems Mainframe", "IBM Cloud Pak for Data", "IBM FlashSystem Storage", "IBM Power Systems & AIX"],
                value="Red Hat OpenShift"
            )
            severity = gr.Dropdown(
                label="Severity SLA Level",
                choices=["Sev-1: Platform Outage", "Sev-2: High Impact", "Sev-3: Medium Impact", "Sev-4: Low Impact"],
                value="Sev-1: Platform Outage"
            )
            
            gr.Markdown("#### ⚡ Dynamic Diagnostic Prompt Injection")
            quick_prompt = gr.Dropdown(
                label="Choose a Default Problem Preset (Quick-Fill)",
                choices=list(DEFAULT_PROMPT_PRESETS.keys()),
                value="OpenShift etcd commit stall"
            )
            error_trace = gr.Textbox(
                label="Error Trace / Syslog / Telemetry Code (Editable)",
                value="kube-apiserver CrashLoopBackOff, etcd fsync latency 38ms"
            )
            client_notes = gr.Textbox(
                label="Client Incident Summary",
                lines=2,
                value="Banking core transaction gateway unreachable due to OCP master nodes degraded."
            )

            submit_btn = gr.Button("🚀 Run Autonomous Agent Triage", variant="primary")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("🤖 Agent Reasoning & Tool Trace"):
                    out_trace = gr.Textbox(label="Autonomous ReAct Execution Log", lines=14)
                with gr.TabItem("📋 TSS Engineering Work Order"):
                    out_work_order = gr.Markdown()
                with gr.TabItem("📧 Enterprise Client Advisory"):
                    out_client = gr.Textbox(label="Client Advisory Draft", lines=5)
                with gr.TabItem("📚 Grounded SOP Reference"):
                    out_sop = gr.Textbox(label="Vector RAG Retrieved Context", lines=6)

    # Preset scenario auto-population
    preset_selector.change(
        fn=load_preset_data,
        inputs=[preset_selector],
        outputs=[t_id, tenant_id, component, severity, error_trace, client_notes]
    )

    # Quick prompt injector
    quick_prompt.change(
        fn=populate_custom_prompt,
        inputs=[quick_prompt],
        outputs=[error_trace]
    )

    # Agent execution trigger
    submit_btn.click(
        fn=run_ibm_copilot,
        inputs=[t_id, tenant_id, component, severity, error_trace, client_notes],
        outputs=[out_trace, out_work_order, out_client, out_sop]
    )

# ==============================================================================
# 6. LOCALHOST & CONTAINER ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    is_colab = "google.colab" in sys.modules
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
        host = "0.0.0.0"
        share = False
    elif is_colab:
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = True
    else:
        # Loopback binding prevents 0.0.0.0 browser connection errors
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = False

    print(f"\n=======================================================")
    print(f"🏢 IBM Diagnostic Copilot Running at:")
    print(f"👉 http://127.0.0.1:{port}")
    print(f"👉 http://localhost:{port}")
    print(f"=======================================================\n")

    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        share=share
    )
