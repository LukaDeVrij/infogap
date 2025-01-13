import pywikibot
import ipdb

def get_item_dict(wikidata_id):
    site = pywikibot.Site("wikidata", "wikidata")
    repo = site.data_repository()
    item = pywikibot.ItemPage(repo, wikidata_id)
    item_dict = item.get()
    return item_dict

def get_popularity(wikidata_id):
    """Get the popularity of the person represented as {wiki_data_id} by 
    counting the number of sitelinks to the person's Wikipedia page.
    """
    item_dict = get_item_dict(wikidata_id)
    popularity = len(item_dict['sitelinks'])
    return popularity

def get_ethnicity(wikidata_id):
    item_dict = get_item_dict(wikidata_id)
    ethnicity_prop = 'P172'
    if ethnicity_prop not in item_dict['claims']:
        return 'unavailable'
    else:
        ethnicity_claim = item_dict['claims'][ethnicity_prop][0]
        ethnicity_entity = ethnicity_claim.getTarget()
        ethnicity = ethnicity_entity.labels['en']
        return ethnicity
    
def get_sex_or_gender(wikidata_id):
    item_dict = get_item_dict(wikidata_id)
    sex_gender_prop = 'P21'
    if sex_gender_prop not in item_dict['claims']:
        return 'unavailable'
    else:
        sex_gender_claim = item_dict['claims'][sex_gender_prop][0]
        sex_gender_entity = sex_gender_claim.getTarget()
        sex_gender = sex_gender_entity.labels['en']
        return sex_gender