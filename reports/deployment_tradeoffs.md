# Deployment Trade-offs: Cloud vs Edge vs On-Prem in Uganda

**System:** XGBoost bank-account-ownership model (`xgb-cw-5ec2d564be13`), served
by the FastAPI container from Tasks 1–2, exported to ONNX in Task 3, and
monitored with prediction logs and PSI in Task 4.
**Data:** household survey answers (age, education, job, marital status,
household size, phone access). This is personal data under Uganda's **Data
Protection and Privacy Act, 2019**, and the output feeds financial-access or
credit-targeting decisions.

## What the model demands, and what it doesn't

Compute is not the deciding factor. The model is 1.5 MB (0.8 MB as ONNX). It
uses about 76 MiB of RAM in its container, needs no GPU, and scores one
respondent in about 0.5 ms with native XGBoost, or about 0.03 ms with ONNX
Runtime, on a single CPU. It needs no external lookups: the 10 survey answers
are all it uses. Any option can run it. What separates the options is
**connectivity, data residency, cost structure, and who keeps it running**.
The last one matters most, because this model was trained on 2016–2018 surveys
and *will* drift. Its Uganda results were already the weakest and least stable
of the four countries (PR-AUC 0.553 ± 0.047), so monitoring and retraining on
Ugandan data are not optional.

## Comparison

| | **Cloud** (hyperscaler) | **Edge** (field officer phone/tablet) | **On-prem / in-country hosting** |
|---|---|---|---|
| **Connectivity outside urban centres** | Every prediction needs a live connection. Rural 3G/4G coverage is patchy and power cuts are common, so outreach stalls exactly where unbanked people are. | Works fully offline; the ONNX model is small and fast enough for a low-end Android device. | Same dependency on connectivity as cloud for field use, but a Kampala server is closer, with lower latency and no international link to fail. |
| **Data sovereignty** | No major hyperscaler has a Ugandan region; the nearest are in South Africa. Storing respondents' data there is a cross-border transfer, which the DPPA only allows under conditions (adequate protection or consent). That needs legal sign-off and adds audit exposure. | Personal data sits on many devices that can be lost or stolen. It needs device encryption, and the model logic can be extracted and "gamed" by applicants. | Data never leaves Uganda: the simplest compliance story for the Personal Data Protection Office (PDPO) and for any Bank of Uganda-supervised partner. |
| **Capex vs opex** | No capex; opex billed in USD, so costs move with the UGX exchange rate. Cheap at this model's scale, but egress and support tiers add up. | Capex on devices (often already owned for mobile-money or survey work); near-zero inference opex; hidden opex in device management. | Capex on a server, UPS/backup power and connectivity, or a monthly fee for rack space in an in-country data centre. One modest CPU server covers this workload many times over. |
| **Who bears maintenance** | Provider runs hardware; the organisation still owns the container, model updates, monitoring and compliance, and needs cloud skills that are scarce and expensive locally. | The organisation, spread across a fleet: pushing model updates to hundreds of devices, version skew, and field support for non-technical users. Heaviest burden. | The organisation's own IT team: patching, backups, power and physical security. Kampala has the skills, but key-person risk is real for small teams. |
| **Fit with what we built** | Container runs unchanged. | Needs the ONNX model plus a port of the input encoder (tested for parity with the Python one), and store-and-forward sync of the JSONL logs, otherwise the Task 4 drift check sees nothing. | Container, log volume and drift job all run unchanged: this is the target Tasks 2 and 4 were built for. |

## Recommendation: in-country hub with offline edge scoring

1. **Hub: on-prem or in-country colocation.** Run the Task 2 container on a
   server in Uganda: the organisation's own machine room or a Kampala data
   centre. It is the system of record, holding the API for connected
   branches and partners, the prediction log volume, and the nightly
   `drift-check --country Uganda` job (exit code 20 raises an alert). This
   keeps personal data in-country, keeps the PSI check on a single log, and
   costs little: one small server with backup power.
2. **Edge: ONNX on field devices, for rural outreach only.** Field officers
   score applicants offline with `model.onnx`. Devices write the same JSONL
   event schema locally (encrypted) and sync it to the hub when connectivity
   returns, so edge predictions enter the same drift monitoring. Devices
   accept only models whose SHA-256 matches the hub's `metadata.json`, and
   they report the running `model_version` on each sync, so outdated
   installs are visible.
3. **Cloud: non-personal workloads only.** CI builds, the container image
   registry and documentation. Personal data, including logs and backups,
   stays in Uganda unless a legal review of the DPPA's cross-border rules
   says otherwise.

**Before go-live, the hub needs what Tasks 1–4 deliberately left out:**
authentication and TLS in front of the API (it currently binds to localhost
only, with no auth), in-country encrypted backups of the log volume, a
retention policy (about 616 bytes per prediction, or about 6 MB per 10,000),
and a named owner for the drift alerts. Retraining on locally collected
Ugandan outcomes is the real long-term fix for the weak Uganda performance.
None of the deployment options solves that on its own.
