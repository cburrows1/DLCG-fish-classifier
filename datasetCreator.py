#Ref: https://medium.com/analytics-vidhya/web-scraping-a-wikipedia-table-into-a-dataframe-c52617e1f451

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bing_image_downloader import downloader
import os
import shutil


# scrape wikipedia table of all fish in florida to create pandas dataframe
wikiurl="https://en.wikipedia.org/wiki/List_of_fishes_of_Florida"
table_class="wikitable sortable jquery-tablesorter"
response=requests.get(wikiurl)

soup = BeautifulSoup(response.text, 'html.parser')
fishtable=soup.find('table',{'class':"wikitable"})

all_td = fishtable.find_all('td')
for td in all_td:
    if td.contents == ['\n']:
        td.string.replace_with('empty')
    for img in td('img'):
        img.replace_with(img['alt'])

df=pd.read_html(str(fishtable))
df=pd.DataFrame(df[0])

# reformat table
data = df.drop(["Scientific name", "Image", "Notes"], axis=1)
data = data.rename(columns={"Common name": "Name", "Freshwater":"Fresh","Saltwater":"salt","Non-native":"nonnative"})

# write to csv
data.to_csv("results/wiki_table.csv")

# collect list of natives and nonnatives from pandas df
native_fish = list(data.query('Native=="check" & salt=="check"')['Name'])
invasive_fish = list(data.query('nonnative=="check" & salt=="check"')['Name'])

FISH_COUNT = 20
search_append = "ocean"

# use bing image search to download FISH_COUNT images of each fish name with the search append, and reformat directory structure.

native_dir = "dataset/natives/"
for i,fish in enumerate(native_fish):
    print("-"*10 + " Downloading native fish {}/{} ".format(i+1,len(native_fish)) + "-"*10)
    search = fish + " " + search_append
    downloader.download(search, limit=FISH_COUNT, output_dir=native_dir, adult_filter_off=True, force_replace=False, timeout=10, verbose=False)

    source = os.path.join(native_dir,search)

    files_list = os.listdir(source)
    for i,files in enumerate(files_list):
        curr_img = os.path.join(source,files)
        ext = os.path.splitext(files)[1]
        dest_path = os.path.join(native_dir, "{}.{}{}".format(fish, i, ext))
        shutil.move(curr_img, dest_path)
    os.rmdir(source)

nonnative_dir = "dataset/nonnatives/"
for i,fish in enumerate(invasive_fish):
    print("-"*10 + " Downloading non-native fish {}/{} ".format(i+1,len(invasive_fish)) + "-"*10)
    search = fish + " " + search_append
    downloader.download(search, limit=FISH_COUNT, output_dir=nonnative_dir, adult_filter_off=True, force_replace=False, timeout=10, verbose=False)

    source = os.path.join(nonnative_dir,search)

    files_list = os.listdir(source)
    for i,files in enumerate(files_list):
        curr_img = os.path.join(source,files)
        ext = os.path.splitext(files)[1]
        dest_path = os.path.join(nonnative_dir, "{}.{}{}".format(fish, i, ext))
        shutil.move(curr_img, dest_path)
    os.rmdir(source)