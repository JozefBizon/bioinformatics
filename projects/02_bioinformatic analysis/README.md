# Project 2 - Bioinformatic Analysis

## Introduction

Imagine you are a bioinformatician working for a clinical genetics department in a large hospital.
Your new task is to analyse sequencing data from a patient with cancer.
You have been provided with raw sequencing data (FASTQ files) from both tumor and normal tissue samples of the patient.
Your goal is to perform a comprehensive bioinformatic analysis to identify somatic mutations, structural variants, and copy number variations that may be driving the cancer.
This analysis will help inform treatment decisions and provide insights into the genetic basis of the patient's disease.


## Objectives

Perform a complete bioinformatic analysis of the provided sequencing data, including the following steps:

1. Raw Data Quality Control: Assess the quality of the raw sequencing data.

2. Read Alignment: Align the sequencing reads to the human reference genome.

3. Postprocessing: Clean and prepare the aligned reads for variant calling.

4. Post-Alignment Quality Control: Evaluate the quality of the aligned reads.

5. Variant Calling: Identify somatic SNVs, indels, structural variants, and copy number variations.

6. Post-Processing and Annotation: Filter and annotate the identified variants, and select those relevant to cancer.

7. Reporting: Summarize the findings in a comprehensive report.


## Submission and Grading Criteria

You will submit a single ZIP file containing:

- Source code of your implementation.

- Annotated data files (VCF, MAF) showing the results of your analysis.

- A brief PDF report (3 - 4 pages) describing your analysis pipeline, interpretation of QC reports, obtained results, and interpretation of the findings.

- A PowerPoint / Keynote presentation (slides) summarizing your analysis and results.


The project will be graded based on the following criteria:

1. Data Processing (**5 points**):

    - Read alignment to reference genome (**2 points**)

    - Postprocessing of aligned reads (**1 point**)

    - Variant calling (**2 points**)

2. Quality Control and Evaluation (**8 points**):

    - Raw data quality assessment (**4 points**)

    - Post-alignment quality assessment (**4 points**)

3. Variant Annotation and Interpretation (**12 points**):

    - Filtering and annotation of variants (**3 points**)

    - Identification of cancer-relevant mutations (**4 points**)

    - Biological interpretation of findings (**5 points**)

4. Presentation of the findings (**20 points**)

**Total: 45 points**

**Minimum passing score: 25 points (56%)**


> **Use of AI is strongly discouraged!**
While it is generally accurate during code generation, it performs poorly in biological data analysis and interpretation, which are key components of this project.
This project is designed to assess your understanding and application of bioinformatic concepts, which AI cannot adequately replicate.
Relying on AI may lead to incorrect or superficial results, ultimately hindering your learning and performance in this course.
**Therefore, the AI may be used for secondary analysis, but the annotation and interpretation must be your own work.**
Use of AI will result in penalties to your grade or failure of the project.


## Further Reading and Resources

- [GATK Best Practices for Somatic Short Variant Discovery](https://gatk.broadinstitute.org/hc/en-us/articles/360035531132-Somatic-short-variant-discovery-SNVs-Indels-)

- All seminars from the course covering relevant tools and methods for each step of the analysis.
