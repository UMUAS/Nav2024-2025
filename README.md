# Nav2024-2025
Navigation Tasks for 2024/2025

## Current tests/observations to be done:

### Navigation:
- [ ] Choose a tracking algorithm that can track objects from a far distance (recommended: at least 30m).
   - [ ] What happens when losing track of an object?
   - [ ] What happens when we choose to track a new object when there is already an object being tracked? Will it override the tracking?
   - [ ] What happens when camera signal is lost?
- [ ] Make sure triangulation calculations are correct when on field. Is it fairly accurate?
- [ ] What happens when multiple processes/calculations are done at the same time (e.g. fire source tracking while IR detecting)? Would any errors stop other processes?
- [ ] Does the KML file report true data based on findings?
   - [ ] Are the GPS positions accurate?
- [ ] What happens if attempted to send KML but failed? Will it try again a few more times before giving up?
- [ ] IR Detection:
   - [ ] What if multiple spots are found? Which spot does the drone prioritize?
   - [ ] What if one spot is found?
   - [ ] What if the same spot is found later? Can it be skipped to avoid a recursive loop?
   - [ ] What if no spots are found?
   - [ ] Detection accuracy?
   - [ ] Influence from sun/reflections? Will the drone try to track them? If so, can the program skip them?
- [ ] Other?

**Please add any other tests/observations that need to be listed.**
