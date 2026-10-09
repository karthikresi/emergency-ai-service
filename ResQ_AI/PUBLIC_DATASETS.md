# Public Dataset Discovery

## Candidate 1
- Dataset name: CrisisNLP
- Source: research benchmark dataset
- License: research use only
- Compatibility: USE_FOR_RESEARCH_ONLY
- Notes: useful for text classification benchmarking, but not a direct replacement for the ResQ hybrid labels.

## Candidate 2
- Dataset name: Disaster Response Messages
- Source: [rmunro/disaster_response_messages](https://github.com/rmunro/disaster_response_messages)
- License: Creative Commons Attribution (per source README)
- Compatibility: EXTERNAL_TEXT_BENCHMARK
- Notes: evaluated using the source-provided splits. It has text plus disaster categories, not ResQ severity, skill, or responder-assignment labels. See the [external validation report](EXTERNAL_VALIDATION_REPORT.md).

## Candidate 3
- Dataset name: Emergency Communications Benchmark
- Source: public benchmark repository
- License: unspecified
- Compatibility: REJECT
- Notes: schema mismatch and unclear label semantics.
