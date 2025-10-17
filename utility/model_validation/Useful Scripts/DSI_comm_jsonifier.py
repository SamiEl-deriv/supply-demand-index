import sys


def jsonify_comms(path):
    """
    JSONify's Metabase DSI commissions for use in dashboards
    """
    with open(path, 'r') as f:
        json_data = f.read()
    formatted = json_data.replace('{', "'{").replace("}", "}'")
    with open('./commissions_DSI.txt', 'w') as f:
        f.write(formatted)

if __name__ == "__main__":
    jsonify_comms(sys.argv[1])