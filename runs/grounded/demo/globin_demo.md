# Grounded GO prediction for globin_demo (147 aa)

Sequence: `MVHFTAEEKAAVTSLWSKMNVEEAGGEALGRLLVVYPWTQRFFDSFGNLSSPSAILGNPKVKAHGKKVLTSFGDAIKNMDNLKPAFAKLSELHCDKLHVDPENFKLLGNVMVIILATHFGKEFTPEVQAAWQKLVSAVAIALAHKYH`

Scores come from HiGO-v2 (fine-tuned ESM2-35M with hierarchy-aware GO queries, recalibrated with domain, ortholog and ligand evidence). The set is closed under the ontology: every listed term's ancestors are also predicted. Only the most specific terms are expanded below.

Domains found by InterProScan:

- IPR050056 Hemoglobin and related oxygen transporters (1-147)
- IPR012292 Globin/Protoglobin (2-147)
- IPR009050 Globin-like superfamily (2-147)
- IPR000971 Globin (3-147)
- IPR002337 Hemoglobin, beta-type (8-146)
- IPR000971 Globin (27-141)
- IPR002337 Hemoglobin, beta-type (30-46)
- IPR002337 Hemoglobin, beta-type (48-63)
- IPR002337 Hemoglobin, beta-type (72-89)
- IPR002337 Hemoglobin, beta-type (94-99)
- IPR002337 Hemoglobin, beta-type (128-144)

## Molecular Function

### GO:0019825 oxygen binding

- Score 0.42 (medium confidence); HiGO before grounding 0.16
- Parent path: GO:0036094 small molecule binding -> GO:0005488 binding -> GO:0003674 molecular_function
- Domain support: IPR012292
- Orthologs: P02100 (Homo sapiens, 100% id, GOA); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, GOA)
- Ligands: O2 (CHEBI:15379) `O=O`
- HiGO evidence residues: P37, T39, N103, G108, H118

### GO:0005515 protein binding

- Score 0.33 (low confidence); HiGO before grounding 0.19
- Parent path: GO:0005488 binding -> GO:0003674 molecular_function
- Orthologs: P02100 (Homo sapiens, 100% id, train label); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, train label)
- HiGO evidence residues: M1, P52, G65, F72, C94

(4 Molecular Function terms in the full set, including ancestors.)

## Biological Process

### GO:0006810 transport

- Score 0.69 (medium confidence); HiGO before grounding 0.05
- Parent path: GO:0051234 establishment of localization -> GO:0051179 localization -> GO:0008150 biological_process
- Domain support: IPR002337
- Orthologs: P02100 (Homo sapiens, 100% id, train label); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, train label)
- HiGO evidence residues: M19, G25, S51, P52, H93

### GO:0008152 metabolic process

- Score 0.19 (low confidence); HiGO before grounding 0.13
- Parent path: GO:0009987 cellular process -> GO:0008150 biological_process
- Orthologs: P68871 (Homo sapiens, 76% id, train label); P02089 (Mus musculus, 73% id, GOA); P02091 (Rattus norvegicus, 73% id, train label)
- HiGO evidence residues: M1, W38, T39, P52, N103

### GO:0032501 multicellular organismal process

- Score 0.15 (low confidence); HiGO before grounding 0.04
- Parent path: GO:0008150 biological_process
- Orthologs: P02100 (Homo sapiens, 100% id, GOA); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, GOA)
- HiGO evidence residues: F4, G57, G65, F123, Y146

### GO:1901700 response to oxygen-containing compound

- Score 0.14 (low confidence); HiGO before grounding 0.11
- Parent path: GO:0042221 response to chemical -> GO:0050896 response to stimulus -> GO:0008150 biological_process
- Orthologs: P68871 (Homo sapiens, 76% id, train label); P02091 (Rattus norvegicus, 73% id, GOA); P69905 (Homo sapiens, 40% id, train label)
- HiGO evidence residues: H3, F4, G57, G65, H147

### GO:0050794 regulation of cellular process

- Score 0.14 (low confidence); HiGO before grounding 0.03
- Parent path: GO:0050789 regulation of biological process -> GO:0065007 biological regulation -> GO:0008150 biological_process
- Orthologs: P02104 (Mus musculus, 82% id, train label); P04444 (Mus musculus, 78% id, train label); P68871 (Homo sapiens, 76% id, GOA)
- HiGO evidence residues: H3, G65, F72, Y146, H147

### GO:0042592 homeostatic process

- Score 0.13 (low confidence); HiGO before grounding 0.04
- Parent path: GO:0008150 biological_process
- Orthologs: P02100 (Homo sapiens, 100% id, GOA); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, GOA)
- HiGO evidence residues: H3, P52, G65, M111, H147

### GO:0048856 anatomical structure development

- Score 0.13 (low confidence); HiGO before grounding 0.02
- Parent path: GO:0032502 developmental process -> GO:0008150 biological_process
- Orthologs: P02100 (Homo sapiens, 100% id, GOA); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, GOA)
- HiGO evidence residues: F4, G65, G120, F123, Y146

### GO:0006979 response to oxidative stress

- Score 0.12 (low confidence); HiGO before grounding 0.07
- Parent path: GO:0006950 response to stress -> GO:0050896 response to stimulus -> GO:0008150 biological_process
- Orthologs: P68871 (Homo sapiens, 76% id, train label); P02091 (Rattus norvegicus, 73% id, GOA); P69905 (Homo sapiens, 40% id, train label)
- HiGO evidence residues: H3, F4, G57, G65, H147

(17 Biological Process terms in the full set, including ancestors.)

## Cellular Component

### GO:0005829 cytosol

- Score 0.68 (medium confidence); HiGO before grounding 0.14
- Parent path: GO:0110165 cellular anatomical structure -> GO:0005575 cellular_component
- Domain support: IPR002337
- Orthologs: P02100 (Homo sapiens, 100% id, train label); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, train label)
- HiGO evidence residues: M1, V21, G65, M79, K83

### GO:0032991 protein-containing complex

- Score 0.58 (medium confidence); HiGO before grounding 0.03
- Parent path: GO:0005575 cellular_component
- Domain support: IPR002337
- Orthologs: P02100 (Homo sapiens, 100% id, train label); P02104 (Mus musculus, 82% id, GOA); P69892 (Homo sapiens, 80% id, train label)
- HiGO evidence residues: M1, P52, H93, M111, W131

### GO:0005576 extracellular region

- Score 0.26 (low confidence); HiGO before grounding 0.28
- Parent path: GO:0110165 cellular anatomical structure -> GO:0005575 cellular_component
- Orthologs: P02100 (Homo sapiens, 100% id, train label); P69892 (Homo sapiens, 80% id, train label); P68871 (Homo sapiens, 76% id, train label)
- HiGO evidence residues: F4, R31, G47, G65, G73

(6 Cellular Component terms in the full set, including ancestors.)
