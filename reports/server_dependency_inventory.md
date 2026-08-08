# Server dependency inventory

Generated 2026-08-07T14:43:38Z for commit `a765b2bca625ff66cf97dc608eacb3a3c63553b5` and tree `7877d29a55c60386ecdde17bb73b717b22d191b8`.

Status: PASS. This inventory records the server/model-runtime closure without
recording credential values. Python 3.11.0rc1 satisfies the repository target
`>=3.11,<3.14`; the declared and required imports are present; `pip check`
passes; and the CUDA runtime reports one H100 MIG device.

Java 17 is installed as OpenJDK `17.0.19+10-1~22.04.2`, including the JDK
compiler `javac`. The missing compiler was the cause of the earlier VnCoreNLP
smoke failure. The exact configured VnCoreNLP resource tree now passes a
non-empty deterministic segmentation smoke. Its observed resource checksum is
`F033BBCF96A1BF27C304F750CDD664D97EB22F1D337C31D7B51BF2DB8693121D`; the
`VnCoreNLP-1.2.jar` SHA-256 is
`9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793`.

The active `azure_rationale_generation` process was not interrupted. No
scientific protocol, model, prompt, dataset, experiment configuration, or
ordering was changed by this environment repair.
