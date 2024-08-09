# Clear everything
rm(list = ls())

# Install and load necessary packages
if (!requireNamespace("bibliometrix", quietly = TRUE)) {
  install.packages("bibliometrix")
}


# Load the package
library(bibliometrix)

biblioshiny()

#
setwd("/Users/seongha/Downloads")

# Combine datasets
library(bibliometrix)

web_data<-convert2df("wos.txt")
scopus_data<-convert2df("scopus.bib", dbsource="scopus", format="bibtex")

combined<-mergeDbSources(web_data, scopus_data, remove.duplicated = T)
library(openxlsx)
write.xlsx(combined, "combined.xlsx")