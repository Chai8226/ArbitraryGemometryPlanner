std_msgs/Header header

uint8 NOMINAL=0
uint8 SAFE_DEGRADED=1
uint8 FAILED=2
uint8 NO_ACTION=3
uint8 outcome

uint8 STAGE_SUCCESS=0
uint8 STAGE_ALREADY_REACHED=1
uint8 STAGE_NOT_READY=2
uint8 STAGE_SHAPE_ASTAR_FAILED=3
uint8 STAGE_OPTIMIZATION_FAILED=4
uint8 stage

uint8 ROUTE_SINGLE_SEGMENT_DIRECT=0
uint8 ROUTE_WHOLE_BODY_REFINED=1
uint8 ROUTE_CERTIFIED_SEED_FALLBACK=2
uint8 ROUTE_NONE=3
uint8 ROUTE_QUALITY_GATE_DIRECT=4
uint8 route

string reason
float64 minimum_interior_speed
float64 maximum_subthreshold_duration
