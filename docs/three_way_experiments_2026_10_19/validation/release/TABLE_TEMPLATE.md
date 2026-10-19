# Three-way experiment report

Preparation and observed outcomes are reported separately. No automatic semantic or poisoning attribution.

| Phase | Frozen proposed budget | Execution approved |
|---|---|---|
| confirm | {'automatic_launch': False, 'registry': None, 'semantic_eta': None} | false |
| evaluate | {'max_components': 2, 'maximum_candidates': 48, 'maximum_responses': 4608, 'mode': 'finite_support_not_certificate', 'pairs_per_component': 24} | false |
| readiness-observation | {'anchors': 4, 'core': 96, 'initial_checks': 32, 'maximum_responses': 132, 'models': ['C', 'LO']} | false |
| readiness-query | {'anchors': 4, 'core': 96, 'maximum_responses': 100, 'models': ['C', 'LQ']} | false |
| seek-main | {'maximum_defender_attempts': 768, 'maximum_discovery_responses': 1536, 'maximum_logical_defender_messages': 384, 'opportunities': 48} | false |
| seek-pilot | {'maximum_defender_attempts': 192, 'maximum_discovery_responses': 384, 'maximum_logical_defender_messages': 96, 'opportunities': 12} | false |
| shield-pilot | {'episodes': 80, 'maximum_defender_responses': None, 'maximum_policy_responses': None} | false |
| utility | {'episodes': 36, 'maximum_defender_responses': None, 'maximum_policy_responses': None} | false |

Clean supported dependencies are not false backdoor certificates. Alarm FPR is N/A without a separate alarm rule.
