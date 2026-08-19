# Kubernetes CPU Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create three Kubernetes YAML manifests that deploy vLLM with the simulated model plugin on CPU-only linux/amd64 nodes without building a custom image.

**Architecture:** An init container installs the `vllm_simulated` Python package into a shared `emptyDir` volume via `pip install --target`. The main container picks it up through `PYTHONPATH`, which makes the `vllm.general_plugins` entry point discoverable. The model `config.json` is mounted from a ConfigMap.

**Tech Stack:** Kubernetes YAML (apps/v1 Deployment, v1 ConfigMap, v1 Service), `vllm/vllm-openai-cpu` Docker image, `python:3.12-slim` init container.

**Spec:** `docs/2026-08-19-kubernetes-cpu-deployment-design.md`

## Global Constraints

- Platform: linux/amd64 (CPU-only nodes, no GPU)
- Main container image: `vllm/vllm-openai-cpu:latest-x86_64` (pin to `v{VERSION}-x86_64` for production)
- Init container image: `python:3.12-slim`
- No custom image build; no GPU resource requests anywhere
- Plugin installed with `--no-deps` (torch/vllm already present in main image)
- All resources in the same namespace (no namespace hardcoded — omit namespace from manifests so the caller controls it with `kubectl apply -n <ns>`)

---

### Task 1: ConfigMap and Service manifests

**Files:**
- Create: `deploy/configmap.yaml`
- Create: `deploy/service.yaml`

**Interfaces:**
- Produces: ConfigMap named `vllm-sim-model-config` with key `config.json`; Service named `vllm-sim` selecting `app: vllm-sim` on port 8000. Task 2's Deployment references both of these names exactly.

- [ ] **Step 1: Create `deploy/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-sim-model-config
data:
  config.json: |
    {
      "architectures": ["SimulatedForCausalLM"],
      "model_type": "qwen3_5",
      "hidden_size": 5120,
      "num_hidden_layers": 64,
      "num_attention_heads": 24,
      "num_key_value_heads": 4,
      "head_dim": 256,
      "vocab_size": 248320,
      "max_position_embeddings": 262144,
      "eos_token_id": 248044,
      "torch_dtype": "bfloat16",
      "latency": {
        "base_ms": 5.0,
        "prefill_ms_per_token": 0.05,
        "decode_ms_per_seq": 1.2,
        "ctx_ms_per_ktoken": 0.3,
        "deterministic_length": true
      }
    }
```

- [ ] **Step 2: Create `deploy/service.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-sim
spec:
  selector:
    app: vllm-sim
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

- [ ] **Step 3: Validate both files**

Run:
```bash
kubectl apply --dry-run=client -f deploy/configmap.yaml
kubectl apply --dry-run=client -f deploy/service.yaml
```

Expected output (each):
```
configmap/vllm-sim-model-config configured (dry run)
service/vllm-sim configured (dry run)
```

If `kubectl` is not available, validate YAML syntax with:
```bash
python3 -c "import yaml; list(yaml.safe_load_all(open('deploy/configmap.yaml'))); print('ok')"
python3 -c "import yaml; list(yaml.safe_load_all(open('deploy/service.yaml'))); print('ok')"
```

- [ ] **Step 4: Commit**

```bash
git add deploy/configmap.yaml deploy/service.yaml
git commit -m "feat: add k8s ConfigMap and Service for vllm-sim"
```

---

### Task 2: Deployment manifest

**Files:**
- Create: `deploy/deployment.yaml`

**Interfaces:**
- Consumes: ConfigMap `vllm-sim-model-config` (from Task 1) mounted at `/model`; Service label `app: vllm-sim` must match `spec.template.metadata.labels.app`
- Produces: Deployment `vllm-sim` — single replica, init container + main container, two volumes (`plugins` emptyDir and `model-config` ConfigMap)

- [ ] **Step 1: Create `deploy/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-sim
  labels:
    app: vllm-sim
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-sim
  template:
    metadata:
      labels:
        app: vllm-sim
    spec:
      initContainers:
      - name: install-plugin
        image: python:3.12-slim
        command:
        - pip
        - install
        - --target=/plugins
        - --no-deps
        - git+https://github.com/vllm-project/vllm-simulated-model.git@main
        volumeMounts:
        - name: plugins
          mountPath: /plugins
      containers:
      - name: vllm
        image: vllm/vllm-openai-cpu:latest-x86_64
        command:
        - vllm
        - serve
        - /model
        - --load-format=dummy
        - --device=cpu
        - --skip-tokenizer-init
        - --gpu-memory-utilization=0.5
        env:
        - name: PYTHONPATH
          value: /plugins
        - name: VLLM_CPU_KVCACHE_SPACE
          value: "4"
        - name: VLLM_CPU_OMP_THREADS_BIND
          value: "0-3"
        ports:
        - containerPort: 8000
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 6
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 3
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
          limits:
            cpu: "4"
            memory: 8Gi
        volumeMounts:
        - name: plugins
          mountPath: /plugins
        - name: model-config
          mountPath: /model
          readOnly: true
      volumes:
      - name: plugins
        emptyDir: {}
      - name: model-config
        configMap:
          name: vllm-sim-model-config
```

- [ ] **Step 2: Validate**

Run:
```bash
kubectl apply --dry-run=client -f deploy/deployment.yaml
```

Expected:
```
deployment.apps/vllm-sim configured (dry run)
```

If `kubectl` is not available:
```bash
python3 -c "import yaml; list(yaml.safe_load_all(open('deploy/deployment.yaml'))); print('ok')"
```

- [ ] **Step 3: Validate all three manifests together**

Run:
```bash
kubectl apply --dry-run=client -f deploy/
```

Expected: three `configured (dry run)` lines, no errors.

- [ ] **Step 4: Commit**

```bash
git add deploy/deployment.yaml
git commit -m "feat: add k8s Deployment for vllm-sim with init container plugin injection"
```

---

### Task 3: Smoke-test on a live cluster (optional but recommended)

Prerequisite: `kubectl` configured against a cluster with CPU nodes, internet egress, and at least 4 CPU / 8 Gi available.

- [ ] **Step 1: Apply all manifests**

```bash
kubectl apply -f deploy/
```

Expected:
```
configmap/vllm-sim-model-config created
service/vllm-sim created
deployment.apps/vllm-sim created
```

- [ ] **Step 2: Watch init container complete**

```bash
kubectl get pod -l app=vllm-sim -w
```

Expected progression: `Init:0/1` → `PodInitializing` → `Running`. Init container typically takes 10–30s depending on network speed.

- [ ] **Step 3: Wait for readiness**

```bash
kubectl wait --for=condition=ready pod -l app=vllm-sim --timeout=120s
```

Expected: `pod/vllm-sim-<hash> condition met`

- [ ] **Step 4: Send a test request**

```bash
kubectl port-forward svc/vllm-sim 8000:8000 &
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 8
  }' | python3 -m json.tool
```

Expected: a JSON response with `choices[0].message.content` containing random tokens and no errors.

- [ ] **Step 5: Tear down**

```bash
kill %1  # stop port-forward
kubectl delete -f deploy/
```

- [ ] **Step 6: Commit if smoke test passed with any adjustments**

```bash
git add deploy/
git commit -m "chore: verify k8s cpu deployment smoke test"
```

---

## Tuning Reference

After the smoke test, adjust these values in `deploy/deployment.yaml` to match your node size:

| What to change | Where | Guidance |
|---|---|---|
| KV cache size | `VLLM_CPU_KVCACHE_SPACE` | increase until ~80% of node memory used |
| Thread binding | `VLLM_CPU_OMP_THREADS_BIND` | match CPU limit, e.g. `0-7` for 8 CPUs |
| Memory cap | `--gpu-memory-utilization` | increase toward 0.9 once stable |
| Plugin version | `@main` in init container command | replace with a commit SHA for production |
| Image version | `latest-x86_64` | replace with `v{VERSION}-x86_64` for production |
