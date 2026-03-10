# Practicum 04: Data Cleaning

## Introduction

This practicum will focus on the cleaning and preprocessing of aligned sequencing data.
You will learn how to use tools such as Samtools and Picard to clean BAM files by removing duplicates, fixing mate information, and recalibrating base quality scores.
This practicum will provide you with hands-on experience in preparing aligned data for downstream analyses such as variant calling.


## Objectives

By the end of this practicum, you will be able to:

1. Inspect mapping quality and alignment metrics using Qualimap.

2. Use Samtools and Picard tools to clean and preprocess BAM files.

3. Remove duplicate reads from BAM files.

4. Recalibrate base quality scores using GATK.

5. Assess the quality of cleaned BAM files.


## Prerequisites

- Bioinformatics resources repository cloned to your local machine. If you haven't done this yet, follow the instructions in the main README file of the repository.

- BAM files aligned by BWA MEM from **Practicum 03: Alignment**.


## Getting Started

Additional software/tools required for this practicum:

- [Samtools](http://www.htslib.org/)

- [Qualimap](http://qualimap.bioinfo.cipf.es/)

- [GATK](https://gatk.broadinstitute.org/hc/en-us)

Most of these tools shoud be already installed if you followed the instructions in the Getting Started section of **Practicum 03: Alignment**, and you can install GATK following these instructions:

```bash
wget https://github.com/broadinstitute/gatk/releases/download/4.4.0.0/gatk-4.4.0.0.zip
unzip gatk-4.4.0.0.zip

cd gatk-4.4.0.0
python3 gatk

# If you want to have gatk command available globally, you can create an alias
nano ~/.bashrc
# Add the following line at the end of the file
alias gatk='python3 /opt/gatk-4.4.0.0/gatk'

# NOTE: /opt/ is the path where the extracted GATK folder is located.
# Replace it with the actual path on your system if GATK is stored elsewhere.

# Reload the bash configuration so the new alias becomes active immediately
source ~/.bashrc
```

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

- [Qualimap Documentation](http://qualimap.bioinfo.cipf.es/doc_html/)

- [GATK Documentation](https://gatk.broadinstitute.org/hc/en-us)



