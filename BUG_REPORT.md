# Bug Report — DataReporter UDP Broadcast

Found during cross-check between `data_report.h` and `visualizer.py`.

---

## Bug 1 (CRITICAL) — UDP silently disabled when file logging is off

**File:** `include/user/data_report.h`

**Problem:**  
`_general_buffer` is only populated inside `if (general_fp)`. The UDP send block
also checks `!_general_buffer.empty()`. This means **UDP broadcast never fires**
if `general_fp` is null — either because `init(false, ...)` was called, or because
`general.txt` failed to open (e.g. no write permission in the working directory).

```cpp
// push_data2buffer — buffer only filled when file is open
if (general_fp) {
    // ... fills _general_buffer
}

// report_data — UDP gated on buffer being non-empty
if (udp_sock_ >= 0 && !_general_buffer.empty()) {
    ...
}
```

**Fix:**  
Decouple buffer population from `general_fp`. Always fill `_general_buffer`,
regardless of whether file logging is enabled:

```cpp
void push_data2buffer(RLController *rlController) {
    std::vector<float> temp;
    _data_report_mutex.lock();

    // Always populate general buffer (needed for UDP even if file logging is off)
    temp.clear();
    for (int i = 0; i < ACT_JOINTS_NUM; ++i) temp.push_back(rlController->joint_act(i));
    for (int i = 0; i < ACT_JOINTS_NUM; ++i) temp.push_back(rlController->joint_pos(i));
    for (int i = 0; i < ACT_JOINTS_NUM; ++i) temp.push_back(rlController->joint_vel(i));
    for (int i = 0; i < ACT_JOINTS_NUM; ++i) temp.push_back(rlController->joint_tau(i));
    for (float i : rlController->base_rpy)      temp.push_back(i);
    for (float i : rlController->base_rpy_rate) temp.push_back(i);
    for (float i : rlController->base_acc)      temp.push_back(i);
    for (float i : rlController->base_quat)     temp.push_back(i);
    _general_buffer = temp;

    // File logging (optional)
    if (general_fp) {
        // _general_buffer already filled above — write it
    }

    // RL buffer (unchanged)
    if (rl_fp) {
        temp.clear();
        for (float i : rlController->action_increment) temp.push_back(i);
        for (float i : rlController->observation)      temp.push_back(i);
        _rl_buffer = temp;
    }

    _data_report_mutex.unlock();
}
```

---

## Bug 2 (MINOR) — `joint_tau` is always zero

**File:** `source/user/custom.cpp` → `RecordMotorState()`

**Problem:**  
`tau_est` is hardcoded to `0.` and never populated from real motor data:

```cpp
ms_tmp.tau_est.at(i) = 0.;  // always zero
```

The torque subplot in the visualizer will be flat. If real torque data is available
from `MotorData`, populate it here.

---

## Data Layout (for reference — confirmed correct)

The 53-float packet order matches the visualizer exactly:

| Slice | Field | Size |
|-------|-------|------|
| `[0:10]` | `joint_act` | 10 |
| `[10:20]` | `joint_pos` | 10 |
| `[20:30]` | `joint_vel` | 10 |
| `[30:40]` | `joint_tau` | 10 |
| `[40:43]` | `base_rpy` | 3 |
| `[43:46]` | `base_rpy_rate` | 3 |
| `[46:49]` | `base_acc` | 3 |
| `[49:53]` | `base_quat` | 4 |
