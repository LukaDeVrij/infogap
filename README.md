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

## V. Scrape the articles you want.   
With the `main_scrape_bios.py` module, you can scrape English articles and their French (`python main_scrape_bios.py scrape-french-bios`) and Russian (`python main_scrape_bios.py scrape-russian-bios`) counterparts. This will scrape all the bios from the LGBTBioCorpus (Park et al., 2021), and store them under `scratch/wiki_bios/`. It shouldn't be too difficult to replace it with the English wikipedia page IDs that you want. By page IDs, I'm referring to the string after `wiki` in `https://en.wikipedia.org/wiki/Gabriel_Attal` (in this case, it is Gabriel_Attal). For an example, I've uploaded Gabriel Attal's English and French biographies (`Gabriel_Attal_en.pkl` and `Gabriel_Attal_fr.pkl`) under `scratch/wiki_bios/`. We will use these as an example. 

## VI. Run the InfoGap pipeline. 
We can run the En<->Fr InfoGap on using `python main_complete_analysis.py execute-complete-gpt`. When you go to the definition of this command, you'll see this code block:

``` 
    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(info_gap_map_dict, 
        {
            'en_bio_id': ["Gabriel_Attal"],
            'fr_bio_id': ["Gabriel_Attal"], 
            'person_name': ["Gabriel Attal"],
            'tgt_person_name': ["Gabriel Attal"]
        },{
        'version': '001'
        }, 
        reduce_info_gaps, 
        'fr_bio_id',
        [BioFilenotFoundError, NoPronounError, ExceptionOOMSingleDataPoint, np.AxisError]
    )
```
Since we're only running it one bio, all of the lists only have one entry. (This looks a bit silly because the en and fr bio IDs are the same, as are the person name entries. They can however all be different, like when we're running it on En<->Ru, instead of En<->Fr). The InfoGap will be computed for all people in this list. Each person's InfoGap is cached under `${SCRATCH_DIR}/full_cache`, thanks to the [flowmason package](https://github.com/smfsamir/flowmason). 

Below this step, you'll also see `map_step_compute_connotations`, which we use for the analyses in Section 3.3 and 3.4 of our [paper](https://arxiv.org/abs/2410.04282). Feel free to exclude this if you're not interested in a sentiment analysis. 

The line `metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache"), full_map_dict, "full_analysis_logs")` executes these steps using the [flowmason](https://github.com/smfsamir/flowmason) package. The line below `info_gap_dfs = load_mr_artifact(metadata[0])` will load the artifact from the [flowmason](https://github.com/smfsamir/flowmason) cache. `info_gap_dfs` is a tuple with three elements (`len(info_gap_dfs)==3`). The first one is the InfoGap for the En->Fr direction, the second for the Fr->En direction. The third element is a historical artifact from earlier in the development, you can safely ignore it. The InfoGaps are stored as `polars` DataFrames. Let's consider `info_gap_dfs[0]` (`En->Fr`) The most important column is `gpt-4_intersection_label`, where `yes` means it is in both `En` and `Fr` while no means it is only in `En`. (Analogous for `info_gap_dfs[1]`, the `Fr->En` direction). 



(NOTE: we only tested it on biographies. On events with complex histories, like border conflicts, it may not be as reliable. At any rate, you'll want to evaluate the results for a couple of samples. More on that below). 

## VII. Evaluating InfoGap on your documents
If you're using this for the first time, you should check that the InfoGap labels are reasonably aligned with your expectations. 

- [ ] TODO: explain how to do the evaluation. 

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
