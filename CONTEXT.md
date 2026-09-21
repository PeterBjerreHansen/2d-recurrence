# Domain context

## Recurrence updates

- `U_T` and `U_D` are the exact temporal and depth update counts for one
  trajectory.
- `update_support` is the ordered set of allowed counts for each axis.
- `update_probabilities` is the joint probability matrix over `(U_T, U_D)`;
  `update_probability_schedule` selects that matrix by absolute optimizer step.
- A concrete `RecurrenceSchedule` contains Boolean write masks, not a pass
  label. The physical number of core passes is derived as
  `max(U_T, U_D) + 1`.
- Update counts control state writes. Physical pass counts describe execution
  cost and must not be used as alternate names for update counts.
