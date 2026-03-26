# State Timeline Failure Breakdown

- input: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/dual_eval/rescored_strict.jsonl`
- variant: `V3_PREFIX_SPLIT`
- target violation: `State timeline is inconsistent at disruption boundary`
- matched rows: 13

## Reason Counts

- UNKNOWN: 10
- IN_TRANSIT_MISMATCH: 2
- TIME_PARSE_FAIL: 1

## Split Mode Counts

- SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE: 13

## Stage Counts

- stage_feedback_v7_13_boundary_test_v8: 13

## Validator Mode Counts

- p8_hard: 13

## Sample Cases

### UNKNOWN

- sample_id: `P8:p8_instance_061` seed=0
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:28` / suffix_first_event_time_before: `15:36` / suffix_first_event_time_after: `15:36`
  - timeline_norm_applied: `1` / shift_min: `178` / overlap_fix: `2`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[5] exceeds photo_time; event[6] exceeds photo_time; event[7] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_062` seed=0
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `13:59` / suffix_first_event_time_before: `14:38` / suffix_first_event_time_after: `14:38`
  - timeline_norm_applied: `0` / shift_min: `0` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_056` seed=1
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:47` / suffix_first_event_time_before: `16:14` / suffix_first_event_time_after: `16:14`
  - timeline_norm_applied: `1` / shift_min: `246` / overlap_fix: `2`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[3] exceeds photo_time; event[4] exceeds photo_time; event[5] exceeds photo_time; event[6] exceeds photo_time; event[7] exceeds photo_time`
- sample_id: `P8:p8_instance_059` seed=1
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:25` / suffix_first_event_time_before: `15:30` / suffix_first_event_time_after: `15:30`
  - timeline_norm_applied: `1` / shift_min: `6` / overlap_fix: `6`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[6] exceeds photo_time; event[7] exceeds photo_time; event[8] exceeds photo_time; event[5] tailor task after close; Immutable past appears modified by disruption terms`
- sample_id: `P8:p8_instance_059` seed=2
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:25` / suffix_first_event_time_before: `15:30` / suffix_first_event_time_after: `15:30`
  - timeline_norm_applied: `1` / shift_min: `230` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[7] exceeds photo_time; event[8] exceeds photo_time; event[9] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_052` seed=4
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:12` / suffix_first_event_time_before: `15:04` / suffix_first_event_time_after: `15:04`
  - timeline_norm_applied: `1` / shift_min: `144` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[6] exceeds photo_time; event[7] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_051` seed=5
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:31` / suffix_first_event_time_before: `15:42` / suffix_first_event_time_after: `15:42`
  - timeline_norm_applied: `1` / shift_min: `216` / overlap_fix: `4`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[4] exceeds photo_time; event[5] exceeds photo_time; event[6] exceeds photo_time; event[7] exceeds photo_time; event[8] exceeds photo_time`
- sample_id: `P8:p8_instance_056` seed=5
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:47` / suffix_first_event_time_before: `16:14` / suffix_first_event_time_after: `16:14`
  - timeline_norm_applied: `1` / shift_min: `288` / overlap_fix: `6`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[5] exceeds photo_time; event[6] exceeds photo_time; event[7] exceeds photo_time; event[8] exceeds photo_time; event[9] exceeds photo_time`
- sample_id: `P8:p8_instance_058` seed=5
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:07` / suffix_first_event_time_before: `14:54` / suffix_first_event_time_after: `14:54`
  - timeline_norm_applied: `1` / shift_min: `134` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[6] exceeds photo_time; event[7] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_059` seed=5
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:25` / suffix_first_event_time_before: `15:30` / suffix_first_event_time_after: `15:30`
  - timeline_norm_applied: `0` / shift_min: `0` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`

### IN_TRANSIT_MISMATCH

- sample_id: `P8:p8_instance_058` seed=2
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:07` / suffix_first_event_time_before: `14:54` / suffix_first_event_time_after: `14:54`
  - timeline_norm_applied: `1` / shift_min: `160` / overlap_fix: `2`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[3] exceeds photo_time; event[4] exceeds photo_time; event[5] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`
- sample_id: `P8:p8_instance_054` seed=5
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:00` / suffix_first_event_time_before: `14:40` / suffix_first_event_time_after: `14:40`
  - timeline_norm_applied: `1` / shift_min: `120` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[7] exceeds photo_time; Immutable past appears modified by disruption terms; State timeline is inconsistent at disruption boundary`

### TIME_PARSE_FAIL

- sample_id: `P8:p8_instance_063` seed=3
  - split_apply_mode: `SYSTEM_CONSTRUCTED_CROSSING_FROM_STATE`
  - next_actionable_time: `14:24` / suffix_first_event_time_before: `13:00` / suffix_first_event_time_after: `13:00`
  - timeline_norm_applied: `1` / shift_min: `504` / overlap_fix: `0`
  - boundary_pre_end_minus_alert_min: `0` / boundary_post_start_minus_alert_min: `0`
  - candidate(route=hotel->church, seg_start=12:50, seg_end=13:20)
  - violations: `event[2] has invalid time format; event[3] exceeds photo_time; State timeline is inconsistent at disruption boundary`

