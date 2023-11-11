from slack_sdk import WebClient

class SlackAPI:
    def __init__(self, token, channel_name):
        self.client = WebClient(token)
        self.channel_id = self.get_channel_id(channel_name)
    
    def get_channel_id(self, channel_name):
        # result = self.client.conversations_list()
        # channels = result.data['channels']
        # channel = list(filter(lambda c: c["name"] == channel_name, channels))[0]
        # channel_id = channel["id"]
        # return channel_id
        pass

    def print(self, text):
        from datetime import datetime
        text = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n' + text
        # result = self.client.chat_postMessage(
        #     channel=self.channel_id,
        #     text=text
        # )
        # return result
        print(text)



token = ""
channel_name = ''
slackBot = SlackAPI(token, channel_name)