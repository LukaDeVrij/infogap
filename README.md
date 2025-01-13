# infogap
Implementation for the InfoGap pipeline will be available soon, definitely by EMNLP'24 (probably sooner). 

# Contact
Email: `fsamir@mail.ubc.ca`

# Artifacts
Our analysis dataframes from Section 3 of our paper are here, in JSON format (about ~600MB each):
1. [En<->Fr](https://www.dropbox.com/scl/fi/oxdphmcxaai2ur7swoz1l/connotation_df_en_fr_flan.json?rlkey=pz82ygv8rx2xybkvv1eaavbo3&st=or1r65no&dl=0)
2. [En<->Ru](https://www.dropbox.com/scl/fi/kavcip55wvbfegaafxy5b/connotation_df_en_ru_mt5.json?rlkey=q7wpn8n6ahwp6xg6vd3g9ogub&st=qw5vvi2z&dl=0)

You can process them with the `polars` package (`pl.read_json(...)`). `pandas` should also work. I recommend inspecting these dataframes before trying out the pipeline on your own documents.  

# Running the pipeline yourself
## I. Install flowmason
1. Clone the repo: `git clone https://github.com/smfsamir/flowmason`
2. Go into directory: `cd flowmason`
3. Checkout the `abstract` branch: `git checkout abstract`
4. Install the package locally `pip install -e .`

## II. Install wikipedia-edit-scrape-tool
1. Clone the repo: `git clone https://github.com/smfsamir/wikipedia-edit-scrape-tool`
2. Go into directory: `cd wikipedia-edit-scrape-tool`
3. Install the package locally `pip install -e .`

## III. Set up an Environment file
The purpose of the `.env` is to set configuration environment variables specific to you. Don't commit this. 
1. In `infogap` project directory, run `touch .env`
2. Create two keys: `SCRATCH_DIR` (where all the artifacts from the pipeline will be stored), and `THE_KEY`, an OpenAI key.



## IV. Install requirements
`pip install -r requirements.txt` (It's possible I missed a couple of modules here, please submit a PR if you find that to be the case and I'll approve right away). 

## V. Scrape the biographies you want 
With the `main_scrape_bios.py` module, you can down

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
