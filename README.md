# infogap
Implementation for the InfoGap pipeline will be available soon, definitely by EMNLP'24 (probably sooner). 

# Contact
Email: `fsamir@mail.ubc.ca`

# Artifacts
Our analysis dataframes from Section 3 of our paper are here, in JSON format (about ~600MB each):
1. [En<->Fr](https://www.dropbox.com/scl/fi/oxdphmcxaai2ur7swoz1l/connotation_df_en_fr_flan.json?rlkey=pz82ygv8rx2xybkvv1eaavbo3&st=or1r65no&dl=0)
2. [En<->Ru](https://www.dropbox.com/scl/fi/kavcip55wvbfegaafxy5b/connotation_df_en_ru_mt5.json?rlkey=q7wpn8n6ahwp6xg6vd3g9ogub&st=qw5vvi2z&dl=0)

You can process them with the `polars` package (`pl.read_json(...)`). `pandas` should also work. I recommend inspecting these dataframes before trying out the pipeline on your own documents.  

# Citation
```
@inproceedings{samir-2024-information,
    title = "Locating Information Gaps and Narrative Inconsistencies Across Languages: A Case Study of LGBT People Portrayals on Wikipedia",
    author = "Samir, Farhan  and
      Park, Chan Young and
      Field, Anjalie and
      Shwartz, Vered and 
      Tsvetkov, Yulia",
    editor = "Al-Onaizan, Yaser and
      Bansal, Mohit and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami",
    publisher = "Association for Computational Linguistics"
}
```
