# Practicum 06: Variant Calling

## Introduction

The aim of this practicum is to introduce you to the process of variant calling from cleaned and preprocessed BAM files.
You will learn how to use popular variant calling tools such as GATK HaplotypeCaller, GATK Mutect2, and Strelka2 for detecting single nucleotide variants (SNVs) and small insertions/deletions (indels).
For structural variant (SV) detection, we will explore tools like Manta and Delly.
This practicum will provide you with hands-on experience in identifying genetic variants, and inspecting and filtering variant calls for downstream analyses.


## Objectives

By the end of this practicum, you will be able to:

1. Understand the principles of variant calling.

2. Use GATK HaplotypeCaller, GATK Mutect2, and Strelka2 for calling SNVs and indels.

3. Explore tools like Manta and Delly for structural variant detection.

4. Inspect and filter variant calls using tools like BCFtools. 


## Prerequisites

- Bioinformatics resources repository cloned to your local machine. If you haven't done this yet, follow the instructions in the main README file of the repository.

- Cleaned and preprocessed BAM files from **Practicum 04: Cleaning**.


## Getting Started

Additional software/tools required for this practicum:

- [Samtools](http://www.htslib.org/)

- [BCFtools](http://www.htslib.org/doc/bcftools.html)

- [GATK](https://gatk.broadinstitute.org/hc/en-us)

- [Strelka2](https://github.com/Illumina/strelka)

- [Manta](https://github.com/Illumina/manta)

- [Delly](https://github.com/dellytools/delly)

- [hap.py](https://github.com/Illumina/hap.py)

Most of these tools shoud be already installed if you followed the instructions in the Getting Started section of **Practicum 04: Data Cleaning**.
You can install additional tools following these instructions (Strelka2, Manta and hap.py require Python 2.7, so they will probably need to be installed in a separate conda environment):

```bash
# Add bctfools and delly to the existing conda environment
conda install -c bioconda bcftools delly

# Create a new conda environment for Strelka2, Manta and hap.py - may not work
conda create -n varcall -c bioconda strelka manta hap.py

# If the installation above does not work, create the environment for variant callers and hap.py separately
conda create -n varcall -c bioconda strelka manta
conda create -n happy -c bioconda hap.py
```

### If you have not done so for the previous practicum, you will also need to download dbSNP and modify the contig names to match those in our reference genome. Otherwise, you can skip the following section.

You should also have the reference genome FASTA file and its index files from **Practicum 03: Alignment**, but you will also need a database of known variants (_dbSNP_) for variant calling with GATK.
Note that the entire database is quite large (15 - 30 GB), so we will use a smaller subset of common variants for this practicum.
Nonetheless, make sure you have enough disk space available (at least 2 GB):

```bash
# Download dbSNP database and its index
wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/common_all_20180418.vcf.gz
wget https://ftp.ncbi.nih.gov/snp/organisms/human_9606/VCF/common_all_20180418.vcf.gz.tbi
```

> **However, there is an issue!** GATK checks whether the contig names in the reference genome and the dbSNP database match, and if they don't, it will throw an error and stop.
The problem is that our FASTA reference genome from UCSC uses the `chr` prefix in contig names (e.g., `chr1`, `chr2`, etc.), while the dbSNP database does not (e.g., `1`, `2`, etc.).
To fix this, we will need to modify the contig names in the dbSNP VCF file to match those in our reference genome.
We can do this using `bcftools annotate` (you can find the `contig_map.txt` in `bioinformatics/06_variant_calling/` directory):

```bash
bcftools annotate --rename-chrs <(awk '{print $1, "chr"$1}' <(contig_map.txt)) \ 
    -O z -o dbsnp.chr.vcf.gz common_all_20180418.vcf.gz

bcftools index -t dbsnp.chr.vcf.gz
```

> This is a common issue in bioinformatics, so you just recieved a taste of the challenges that can arise when working with different datasets and tools :).


## Further Reading and Resources

- [Samtools Documentation](http://www.htslib.org/doc/samtools.html)

- [BCFtools Documentation](http://www.htslib.org/doc/bcftools.html)

- [GATK Documentation](https://gatk.broadinstitute.org/hc/en-us)



