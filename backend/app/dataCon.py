class dataCon:

    def __init__(self, db):
        self.dbaseConn = db

    def save_text(self, inputTxt):
        saveQry = "INSERT INTO syllabotData (data) VALUES (%s)"
        try:
            tcursor = self.dbaseConn.db.cursor()
            tcursor.execute(saveQry, [inputTxt])
            self.dbaseConn.db.commit()
            return True
        except Exception as e:
            print(e)
            self.dbaseConn.db.rollback()
            return False
