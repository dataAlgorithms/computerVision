#!/usr/bin/env python
#!coding=utf-8

import psycopg2
import json
import requests

# Connect to postgresql
def connectSql(database="deepface_cluster", user="postgres", password="postgres", host="127.0.0.1", port="5432"):

    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
    return conn

# Query for postgresql
def querySql(conn=None, query="select * from ranker_candidate limit 10"):

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return rows

# Delete records
def deleteSql(conn=None, query="delete from ranker_candidate"):

    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    conn.close()

# Clear sql and fse
def clearSqlFse():
    # clear fse
    repo = Ranker2Repo("http://192.168.2.16:8010/rank/repo")

    # Do the repo query operation
    print '\r::Do the repo query'
    query_source = {"Context":{},"Repo":{"Operation":4}}
    response = repo.queryRepo(query_source)
    print('response:', response)

# Post request
def post_request(url, source):
    jsource = json.dumps(source)
    print('jsource:', jsource)
    resp = requests.post(url, data = jsource)

    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

# Ranker
class Ranker2Repo:
    def __init__(self, url):
        self.url = url

    def addRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        print '::Add repo result is as follow!'
        print 'resp: ', json.dumps(resp_dict, indent=1)

    def queryRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Query repo result is as follow!'
        print 'resp: ', json.dumps(resp_dict, indent=1)
        return resp_dict

    def deleteRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Delete repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def updateRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::update repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)    

# Sax
def startSax():

    pass

# Fse
def startFse():
    
    pass

# Matrix
def startMatrix():

    pass

if __name__ == '__main__':
    clearSqlFse()
