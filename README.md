# Contact
Paper: https://aclanthology.org/2024.emnlp-main.384/

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
If you're using this for the first time, you should definitely check that the InfoGap labels are reasonably aligned with your expectations. This is what the final two steps are for:

### Preparing the annotation frame
```
full_map_dict['step_prep_annotation_frame'] = SingletonStep(step_prep_annotation_frame, { # samples 10 facts from the InfoGap frame for each direction (20 in total)
    'info_gap_dfs': 'map_step_compute_info_gap', 
    'tgt_lang_code': 'fr', 
    'intersection_label': 'gpt-4_intersection_label',
    'version': '003'
})
full_map_dict['step_add_annotation_translations'] = SingletonStep(step_add_translations_to_annotation_frame, { # adds translations for {tgt_lang_code} using NLLB-200, in case you don't read {tgt_lang_code}
    'annotation_frame': 'step_prep_annotation_frame', 
    'target_fname': 'attal_annotation_frame.json',
    'version': '001'
})
```
### Performing annotations
This will result in a JSON file that will store annotations; in this case, it is `attal_annotation_frame.json`, since we're using Gabriel Attals `En` and `Fr` pages as the running example. Then, running `python main_perform_annotation.py` should result in the following output in your terminal: 

```
2025-01-27 15:29:30.559 | INFO     | packages.annotate:annotate_frame:72 - Number of samples that are unannotated: 20
  0%|                                                                                            | 0/10 [00:00<?, ?it/s]

Consider the following fact(s) about Gabriel Attal:

1. The French media speculated that Attal was a potential contender in the 2027 presidential election.
2. On 16 January 2024, Attal made an announcement.
3. Attal announced that he would not be seeking a vote of confidence in the National Assembly.


Is the final fact present in the French Wikipedia article about Gabriel Attal (fr.wikipedia.org/wiki/Gabriel_Attal)?

Here are some snippets from the French article:
1. Emmanuel Macron a annoncé la dissolution de l'Assemblée le soir des élections européennes. (Emmanuel Macron announced the dissolution of the Assembly on the eve of the European elections.)
2. Gabriel Attal n'a pas été consulté avant l'annonce de la dissolution de l'Assemblée. (Gabriel Attal was not consulted before the dissolution of the Assembly was announced.)

1. Le 8 juillet 2024, Gabriel Attal remet sa démission et celle de son gouvernement au président de la République. (On 8 July 2024, Gabriel Attal submitted his resignation and that of his government to the President of the Republic.)
2. Le président de la République refuse la démission de Gabriel Attal. (The President of the Republic refuses the resignation of Gabriel Attal.)


A: covered by the snippets
B: partly covered by the snippets
C: covered by the article
D: partly covered by the article
E: Not in the article
Answer (A/B/C/D/E):
```
How it works:
- You read the source facts at the beginning. We provide up to two facts of previous context, but the fact of interest is the final one. In particular, whether that fact exists in the other language version.
- Suppose it does exist in the other article:
    - In this case, you will pick either A, B, C, D
    - You pick A or B when the target fact is shown in the small set of snippets from the other language version
    - Otherwise you pick C or D. To select C or D, you'll have to go through the other language version's article directly on Wikipedia and see if you can find the fact in there.
- Otherwise, you pick E. 


A few things to note:
- You can see at the start of the annotation, the total number of samples in the annotation JSON that haven't been annotated (at the beginning this will be 20).
- A progress bar that says 0/10. This may be confusing because there are 20 samples to be annotated. This is because I try to annotate 10 samples per sitting (each annotation is not easy since you may have to read the target article in full to see whether the fact is listed/inferrable or not. You can annotate more than 10 in one sitting by changing the `num_samples` parameter in the call to `annotate_frame` in `main_perform_annotation.py`.
- When you finish all 10 (for the sitting), or Ctrl+C and exit, your annotations will be saved. Next time you run the annotation, the terminal output will show that you have `20-n` annotations to complete.
- It's also instructive to read Section 2.3 of the [paper](https://arxiv.org/pdf/2410.04282) to understand the terminal content for each datapoint. 



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
