DESC='''Preparing publications list to be uploaded to BI proceedings.

By: AA
'''

import argparse
import pandas as pd
from pdb import set_trace

from consts import *

def author_format(auth_list):
    # Assumes author formate is Surname, name1 name2
    mod_auth_list = []
    for auth in auth_list.split('; '):
        if auth == '':
            continue
        words = auth.split(', ')
        initials = []
        for word in words[1].split(' '):
            initials.append(word[0])
        initials_str = ' '.join(initials)
        mod_auth_list.append(words[0] + ' ' + initials_str)

    return '; '.join(mod_auth_list)

def main():

    # read csv file from google scholar
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_file", 
            help="Input publications list in csv form from Google Scholar.")
    
    args = parser.parse_args()

    pubs = pd.read_csv(args.input_file)

    # change author name format to "Doe J"
    pubs.Authors = pubs.Authors.apply(author_format)

    # change publication date format

    # column names
    pubs = pubs.rename(columns={
        'Publication': 'Publication Name',
        'Volume': 'Publication Vol Number'})
    for col in ['Publication Date', 'Artifact Type', 'Short Title',
            'Abstract', 'Contribution to Science', 'Rights (3rd Party)',
            'Rights (Grants and Contracts)', 'External Link(URL)',
            'Author Affiliations', 'Peer Reviewed', 
            'Location of Publisher/Conference', 'Month', 'Day', 'DOI',
            'PMCID', 'PMID', 'Citation', 'Funding Agency Award Number',
            'Funding Agency Report Number', 'UVA Reference Number(s)',
            'Division', 'Pre UVA BI Artifact', 'Former Technical Report Number',
            'Authored On', 'Patent Number', 'Inventor Name', 'Assignee',
            'Method', 'Patent Method', 'Profiles', 'Associated Projects']:
        pubs[col] = ''

    # add proceedings columns with values if possible

    # nan
    pubs = pubs.fillna('')

    # write
    pubs.to_csv('publications_to_proceedings.csv', index=False)

if __name__ == "__main__":
    main()
