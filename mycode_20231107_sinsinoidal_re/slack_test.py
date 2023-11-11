from slack_sdk import WebClient

class SlackAPI:
    """
    슬랙 API 핸들러
    """
    def __init__(self, token, channel_name):
        # 슬랙 클라이언트 인스턴스 생성
        self.client = WebClient(token)
        self.channel_id = self.get_channel_id(channel_name)
    
    def get_channel_id(self, channel_name):
        """
        슬랙 채널ID 조회
        """
        # conversations_list() 메서드 호출
        result = self.client.conversations_list()
        # 채널 정보 딕셔너리 리스트
        channels = result.data['channels']
        # 채널 명이 'test'인 채널 딕셔너리 쿼리
        channel = list(filter(lambda c: c["name"] == channel_name, channels))[0]
        # 채널ID 파싱
        channel_id = channel["id"]
        return channel_id


    def print(self, text):
        """
        슬랙 채널에 메세지 보내기
        """
        from datetime import datetime
        text = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n' + text
        # chat_postMessage() 메서드 호출
        result = self.client.chat_postMessage(
            channel=self.channel_id,
            text=text
        )
        return result



token = "xoxb-207841948934-5240259081345-JYsQQiuBQcYmo1MPjWKm8oOg"
channel_name = 'cikm-traffic-experiment-report'
slackBot = SlackAPI(token, channel_name)
#slackBot.post_message('test')

#query = "슬랙 봇 테스트"
#text = "자동 생성 문구 테스트"

# 채널ID 파싱
#channel_id = slack.get_channel_id(channel_name)
# 메세지ts 파싱
#message_ts = slack.get_message_ts(channel_id, query)
# 댓글 달기
#slack.post_thread_message(channel_id, message_ts, text)
#slack.post_message(channel_id, text)
