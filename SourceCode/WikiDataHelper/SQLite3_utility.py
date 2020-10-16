import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def if_entity_exist_search(conn, input):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param input:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM Toponyms WHERE NAME=?", (input,))

    rows = cur.fetchall()

    if len(rows) >= 1:
        return True

    return False


def insert_record(conn, input_tuple):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ' INSERT INTO Toponyms (ID, NAME) VALUES (?, ?); '
    cur = conn.cursor()
    cur.execute(sql, input_tuple)
    conn.commit()
    cur.close()
    return cur.lastrowid

