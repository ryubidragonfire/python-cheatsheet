import configparser

if __name__ == '__main__':

    """
    Parsing configurations
    """
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    SECRET_KEY = config['DEFAULT']['SECRET_KEY']; print(SECRET_KEY)
    SECRET_KEY = config['SECONDARY']['SECRET_KEY']; print(SECRET_KEY)