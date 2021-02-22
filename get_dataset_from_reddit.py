from contextlib import closing

import praw
import psaw
import sqlite3
import progressbar


def not_in(db, id):
    with closing(db.cursor().execute('SELECT id from comments where id = ?', (id,))) as cursor:
        if cursor.fetchone():
            return False
        return True


def main():
    reddit = praw.Reddit(client_id="jSTLDT5NQzi6LA",
                         client_secret="Q2VODbHrd_Zykjj0zcWi0z7M3MA",
                         password="Malkaleet13",
                         user_agent="markover",
                         username="SlipSaleEtOdorant")
    api = psaw.PushshiftAPI(reddit)
    db = sqlite3.connect('r_france.sqlite')
    gen = api.search_comments(subreddit='france')
    for comment in progressbar.progressbar(gen):
        if comment.author is None:
            author = ''
        else:
            author = comment.author.name
        if not_in(db, comment.id):
            db.cursor().execute('INSERT INTO comments VALUES(?, ?, ?, ?)',
                                (comment.id, author, comment.body, comment.created_utc)).close()
            db.commit()



if __name__ == '__main__':
    main()
