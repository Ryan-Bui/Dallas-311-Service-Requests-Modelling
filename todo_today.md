label,count
[ServiceRequest],14989
[Location],13992
[ServiceRequestType],260
[DocumentChunk],144
[Outcome],39
[Department],26
[CityCouncilDistrict],18
[MethodReceived],7
[Status],7
[Priority],6
[DocumentSource],2

nodes: 29490
relationships: 87907

What The Circle Means
In a force-directed layout, a perfect circle forms when every node has nearly identical degree and there are no clusters. Your graph has no hierarchy, no community structure, no hubs of meaningful density. Everything is uniformly connected at the same shallow depth.
29,490 nodes  ÷  87,907 relationships  =  ~3.0 avg degree

Sounds healthy — but the circle says it's fake density.

What Almost Certainly Happened
Your extractor ignored the rule to drop Service Request Numbers and created one node per 311 record:
# What got ingested:
(SR_123456)  -[LOCATED_IN]->   (District 4)
(SR_123457)  -[LOCATED_IN]->   (District 4)
(SR_123458)  -[LOCATED_IN]->   (District 4)
... × 29,000 more

# Every pink dot = one service request record
# Every teal dot = one of your ~14 real domain entities
# District 4 has 2,000+ edges — but they're all meaningless LOCATED_IN to records
The Correct 311 Graph Model
311 data is tabular event data — individual requests should never be nodes. They are the raw material for extracting patterns, not entities themselves.
WRONG model (what you built):
(SR_00123) → (District4)
(SR_00124) → (District4)        29,000 request nodes
(SR_00125) → (District4)        all degree 1-3
...

CORRECT model (what you need):
(PotholeRepair) -[RECURS_IN {count:847, avg_days:4.2}]→ (District4)
(StreetServices) -[EXCEEDS_ERT {rate:0.34, type:"Pothole"}]→ (District4)
(District4) -[OVERLOADS {request_volume:2847}]→ (StreetServices)
The 311 records are the evidence for relationships — not the relationships themselves. Each row in your CSV should contribute to enriching an edge property, not create a new node.