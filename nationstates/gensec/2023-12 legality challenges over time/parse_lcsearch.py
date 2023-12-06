#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:53:42 2023
@author: Imperium Anglorum
"""
import re
from glob import glob
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

html = []
for g in sorted(glob('lc-*.html')):
    with open(g) as f:
        html.append(f.read())

frames = []
for page in html:

    soup = BeautifulSoup(page, 'lxml')
    for post in soup.select('div.post'):
        author = post.select('dt.author a')[0].text
        date = post.select('dl.postprofile dd')[0].text
        title = post.select('div.postbody h3 a')[0].text
        url = post.select('div.postbody h3 a')[0]['href']

        frames.append({
            'author': author,
            'date': date,
            'title': title,
            'url': url
        })

lc = pd.DataFrame(frames)
lc['date'] = lc['date'].pipe(pd.to_datetime)
lc['post'] = lc['url'].str.extract(r'&p=(\d+)(?:&|$)', expand=False)
lc.drop_duplicates('post', keep='last', inplace=True)

mask = lc['title'].str.contains(
        r'^\[((legal(ity)? )?challenge|LITIGATION|legality)\]', regex=True,
        flags=re.IGNORECASE) | \
    lc['title'].str.contains(
        r'^\[.*?challenge.*?\]', regex=True,
        flags=re.IGNORECASE) | \
    lc['title'].str.contains(
        r'^(legality|sua sponte|at vote)? ?(legality)? ?challenge:', 
        regex=True, flags=re.IGNORECASE) | \
    lc['title'].str.contains(
        r'^\(Challenge.+?\)', regex=True,
        flags=re.IGNORECASE)
lc = lc[mask].sort_values('date')
lc.to_csv('lc_search_list_20231205.csv.xz', index=False)

# ----------------------------------------------------------------------------
# legality challenges brought by quarter

f, ax = plt.subplots(figsize=(8 ,4))
lc.groupby(lc['date'].round('D') + pd.offsets.QuarterEnd(0))['title'] \
    .count().plot(ax=ax)
    
ax.set_title('Legality challenges brought forumside by quarter')
f.savefig('lc_counts.jpeg', bbox_inches='tight')

# ----------------------------------------------------------------------------
# legality challenges by author

f1, ax1 = plt.subplots(figsize=(8, 6))

lc['author'].value_counts().head(20).sort_values().plot.barh(ax=ax1)

ax1.set_title('Legality challenges brought by author')
ax1.set_ylabel(None)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax1.grid(True, linestyle='dotted')

f1.savefig('lc_authors.jpeg', bbox_inches='tight')