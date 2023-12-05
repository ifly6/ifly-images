#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:53:42 2023
@author: Imperium Anglorum
"""
import re

import pandas as pd
from glob import glob
from bs4 import BeautifulSoup

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
