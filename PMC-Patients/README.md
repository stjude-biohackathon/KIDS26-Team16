---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- patient summary
- medical
- biology
size_categories:
- 100K<n<1M
---

# Dataset Card for PMC-Patients

## News
We released PMC-Patients-V2 (in JSON format with the same keys), which is based on 2024 PMC baseline and contains 250,294 patients.
The data collection pipeline remains the same except for using more PMC articles.

## Dataset Description

- **Homepage:** https://github.com/pmc-patients/pmc-patients
- **Repository:** https://github.com/pmc-patients/pmc-patients
- **Paper:** https://arxiv.org/pdf/2202.13876.pdf
- **Leaderboard:** https://pmc-patients.github.io/
- **Point of Contact:** zhengyun21@mails.tsinghua.edu.cn

### Dataset Summary

**PMC-Patients** is a first-of-its-kind dataset consisting of 167k patient summaries extracted from case reports in PubMed Central (PMC), 3.1M patient-article relevance and 293k patient-patient similarity annotations defined by PubMed citation graph.


### Supported Tasks and Leaderboards

**This is purely the patient summary dataset with relational annotations. For ReCDS benchmark, refer to [this dataset](https://huggingface.co/datasets/zhengyun21/PMC-Patients-ReCDS)**

Based on PMC-Patients, we define two tasks to benchmark Retrieval-based Clinical Decision Support (ReCDS) systems: Patient-to-Article Retrieval (PAR) and Patient-to-Patient Retrieval (PPR). 
For details, please refer to [our paper](https://arxiv.org/pdf/2202.13876.pdf) and [leaderboard](https://pmc-patients.github.io/).

### Languages

English (en).

## Dataset Structure

### PMC-Paitents.csv

This file contains all information about patients summaries in PMC-Patients, with the following columns:

- `patient_id`: string. A continuous id of patients, starting from 0.
- `patient_uid`: string. Unique ID for each patient, with format PMID-x, where PMID is the PubMed Identifier of the source article of the patient and x denotes index of the patient in source article.
- `PMID`: string. PMID for source article.
- `file_path`: string. File path of xml file of source article.
- `title`: string. Source article title.
- `patient`: string. Patient summary.
- `age`: list of tuples. Each entry is in format `(value, unit)` where value is a float number and unit is in 'year', 'month', 'week', 'day' and 'hour' indicating age unit. For example, `[[1.0, 'year'], [2.0, 'month']]` indicating the patient is a one-year- and two-month-old infant.
- `gender`: 'M' or 'F'. Male or Female.
- `relevant_articles`: dict. The key is PMID of the relevant articles and the corresponding value is its relevance score (2 or 1 as defined in the ``Methods'' section).
- `similar_patients`: dict. The key is patient_uid of the similar patients and the corresponding value is its similarity score (2 or 1 as defined in the ``Methods'' section).

## Dataset Creation

If you are interested in the collection of PMC-Patients and reproducing our baselines, please refer to [this reporsitory](https://github.com/zhao-zy15/PMC-Patients).

### Citation Information

If you find PMC-Patients helpful in your research, please cite our work by:

```
@article{zhao2023large,
  title={A large-scale dataset of patient summaries for retrieval-based clinical decision support systems},
  author={Zhao, Zhengyun and Jin, Qiao and Chen, Fangyuan and Peng, Tuorui and Yu, Sheng},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={909},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```