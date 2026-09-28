# data/grounding

Built by `PYTHONPATH=src python -m cafa6.grounding_data --step all` (README generated 2026-09-28 14:43). Re-running skips existing outputs.

**Leakage rule:** in `uniprot_features.parquet`, `binding_chebi`, `cofactor_chebi` and `ec` are curated UniProt annotations — use them ONLY for training/reference proteins. `interpro`/`pfam` are computational matches and may be used for any protein.

## Products

| file | size | rows | description |
|---|---|---|---|
| `chebi.tsv` | 2.06 MB | 15251 | [chebi, name, smiles, smiles_from] ChEBI ids in go2chebi or Rhea participants; SMILES from Rhea, else ChEBI structures; class terms without structure borrow the is_a descendant (depth<=3) used in most Rhea reactions (smiles_from=descendant:...; parent:... for secondary ids) |
| `ec.tsv` | 0.38 MB | 8472 | [ec, name, transferred_to] ExPASy ENZYME |
| `ec2go.tsv` | 0.10 MB | 4831 | [ec, go] from GO external2go/ec2go |
| `go2chebi.tsv` | 1.71 MB | 27398 | [go, chebi, relation, axiom] CHEBI references in go-plus logical definitions / restrictions (all aspects) |
| `go2ligand.tsv` | 13.89 MB | 73350 | [go, go_name, chebi, name, smiles, source, relation, via, ubiquitous, smiles_from] MF terms only; source go_logical_def|rhea|ec; ubiquitous = ChEBI in >=300 Rhea reactions (H2O, H+, ATP, NAD+, ...) |
| `goa_reference.parquet` | 11.13 MB | 2349235 | [id, go, evidence, date, taxon, aspect] GOA for 10 reference proteomes, all evidence codes, NOT-qualified rows dropped, DB=UniProtKB only |
| `goa_summary.json` | 0.00 MB |  | per-proteome GOA row counts |
| `interpro2go.tsv` | 0.63 MB | 30122 | [interpro, go] from GO external2go/interpro2go |
| `rhea2go.tsv` | 0.17 MB | 7875 | [rhea, go] from GO external2go/rhea2go |
| `rhea_participants.tsv` | 2.03 MB | 87619 | [rhea (master id), chebi] Rhea REST export (all reaction participants) |
| `uniprot_coverage.json` | 0.00 MB |  | fraction of accessions found / with >=1 InterPro etc. per set |
| `uniprot_features.parquet` | 7.53 MB | 285414 | [id, interpro, pfam, ec, binding_chebi, cofactor_chebi, taxon, found, primary_acc, feature_source(uniprotkb|uniprotkb_secondary|uniparc|none)] UniProt REST for CAFA train, GO-GPT and Kaggle-superset accessions; deleted entries get InterPro/Pfam from UniParc (sequence MD5) |

## Raw sources (`raw/`)

| file | url | downloaded | stored size |
|---|---|---|---|
| `18.E_coli_MG1655.goa.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/18.E_coli_MG1655.goa | 2026-09-28 | 0.92 MB |
| `30.M_tuberculosis_ATCC_25618.goa.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/30.M_tuberculosis_ATCC_25618.goa | 2026-09-28 | 0.43 MB |
| `78.S_pombe.goa.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/78.S_pombe.goa | 2026-09-28 | 1.51 MB |
| `chebiId_name.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/chebiId_name.tsv | 2026-09-28 | 0.14 MB |
| `chebi_compounds.tsv.gz` | https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/compounds.tsv.gz | 2026-09-28 | 6.99 MB |
| `chebi_relation.tsv.gz` | https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/relation.tsv.gz | 2026-09-28 | 2.70 MB |
| `chebi_relation_type.tsv.gz` | https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/relation_type.tsv.gz | 2026-09-28 | 0.00 MB |
| `chebi_structures.tsv.gz (streamed)` | https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/structures.tsv.gz | 2026-09-28 | 0.00 MB |
| `ec2go.gz` | http://current.geneontology.org/ontology/external2go/ec2go | 2026-09-28 | 0.07 MB |
| `enzyme.dat.gz` | https://ftp.expasy.org/databases/enzyme/enzyme.dat | 2026-09-28 | 3.00 MB |
| `go-plus.json.gz` | http://purl.obolibrary.org/obo/go/extensions/go-plus.json.gz | 2026-09-28 | 8.88 MB |
| `go.obo.gz` | http://purl.obolibrary.org/obo/go.obo | 2026-09-28 | 4.88 MB |
| `goa_arabidopsis.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/ARABIDOPSIS/goa_arabidopsis.gaf.gz | 2026-09-28 | 5.55 MB |
| `goa_fly.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/FLY/goa_fly.gaf.gz | 2026-09-28 | 3.36 MB |
| `goa_human.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz | 2026-09-28 | 11.02 MB |
| `goa_mouse.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/MOUSE/goa_mouse.gaf.gz | 2026-09-28 | 9.74 MB |
| `goa_rat.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/RAT/goa_rat.gaf.gz | 2026-09-28 | 7.79 MB |
| `goa_worm.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/WORM/goa_worm.gaf.gz | 2026-09-28 | 2.65 MB |
| `goa_yeast.gaf.gz` | https://ftp.ebi.ac.uk/pub/databases/GO/goa/YEAST/goa_yeast.gaf.gz | 2026-09-28 | 2.14 MB |
| `interpro2go.gz` | http://current.geneontology.org/ontology/external2go/interpro2go | 2026-09-28 | 0.51 MB |
| `rhea-chebi-smiles.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea-chebi-smiles.tsv | 2026-09-28 | 0.17 MB |
| `rhea-directions.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea-directions.tsv | 2026-09-28 | 0.16 MB |
| `rhea-ec-iubmb.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea-ec-iubmb.tsv | 2026-09-28 | 0.04 MB |
| `rhea-relationships.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea-relationships.tsv | 2026-09-28 | 0.09 MB |
| `rhea2ec.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea2ec.tsv | 2026-09-28 | 0.06 MB |
| `rhea2go.gz` | http://current.geneontology.org/ontology/external2go/rhea2go | 2026-09-28 | 0.10 MB |
| `rhea2go.tsv.gz` | https://ftp.expasy.org/databases/rhea/tsv/rhea2go.tsv | 2026-09-28 | 0.06 MB |
| `rhea_reactions_chebi.tsv.gz` | https://www.rhea-db.org/rhea/?query=&columns=rhea-id%2Cchebi-id%2Cequation&format=tsv&limit=1000000 | 2026-09-28 | 0.55 MB |

Total size of data/grounding: 137.0 MB.

## Notes / substitutions

- GO CHEBI links come from `go-plus.json.gz` (plain `go.obo` carries no CHEBI logical definitions); `go.obo` is used only for names/namespaces. purl.obolibrary.org is tried first, current.geneontology.org is the mirror.
- BRENDA requires an account: only a stub (env BRENDA_EMAIL/BRENDA_PASSWORD); the EC route uses Rhea (rhea2ec + rhea-ec-iubmb) and ExPASy ENZYME transferred-entry resolution instead. Partial ECs (x.x.x.-) are not expanded.
- Rhea participants include currency metabolites; filter with `ubiquitous` downstream, but only for source rhea/ec (a go_logical_def row such as ATP binding -> ATP is also flagged ubiquitous and is the actual ligand).
- `chebi.tsv` is restricted to go2chebi + Rhea ChEBI ids; some UniProt binding/cofactor ChEBI ids are not in it.
- UniProt merged accessions are resolved via `sec_acc:` search (`primary_acc` column); deleted ones have found=False, empty curated lists, and InterPro/Pfam from UniParc (matched by sequence MD5; UniParc lists only InterPro entries attached to member-database signatures, so it can be slightly sparser than UniProtKB xrefs).
- GOA: human/mouse/rat/arabidopsis/yeast(559292)/fly/worm from species GAFs; S. pombe, E. coli MG1655 (K-12) and M. tuberculosis ATCC 25618 (H37Rv) from goa/proteomes. `taxon` is the requested reference taxon id.

## UniProt coverage

`found` = in current UniProtKB (incl. merged accessions); `>=1 InterPro` includes UniParc fill-in for deleted entries; `InterPro (UniProtKB only)` excludes it.

| set | n | found | >=1 InterPro | InterPro (UniProtKB only) | >=1 Pfam | EC | binding ChEBI | cofactor ChEBI |
|---|---|---|---|---|---|---|---|---|
| cafa_train | 65603 | 1.000 | 0.983 | 0.983 | 0.967 | 0.339 | 0.305 | 0.139 |
| cafa_val | 8377 | 1.000 | 0.986 | 0.986 | 0.971 | 0.338 | 0.306 | 0.145 |
| cafa_test | 8058 | 1.000 | 0.980 | 0.980 | 0.959 | 0.332 | 0.304 | 0.144 |
| gogpt_train | 119749 | 0.984 | 0.959 | 0.944 | 0.931 | 0.288 | 0.234 | 0.122 |
| gogpt_val | 13305 | 0.984 | 0.958 | 0.943 | 0.930 | 0.291 | 0.234 | 0.121 |
| gogpt_test | 8630 | 0.642 | 0.979 | 0.629 | 0.958 | 0.173 | 0.126 | 0.078 |
| kaggle_superset | 224309 | 1.000 | 0.952 | 0.952 | 0.932 | 0.338 | 0.291 | 0.151 |
