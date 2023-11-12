import requests


class FileDownloader:


    def __init__(self):
        pass


    #region Public Static Methods

    @staticmethod
    def download( url, dest):

        response = requests.get(url, allow_redirects=True)

        with open(dest, "wb") as f:
            f.write( response.content)


    #endregion