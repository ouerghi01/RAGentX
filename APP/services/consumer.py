
import json
import time
import requests
from kafka import KafkaConsumer

POWER_BI_URL = "https://api.powerbi.com/beta/dbd6664d-4eb9-46eb-99d8-5c43ba153c61/datasets/64df15a4-0bcc-4821-a00e-236bdfa843d2/rows?redirectedFromSignup=1%2C1&experience=power-bi&key=%2FTGqQVMjJUF%2B%2Fb8PVJX%2FqyR%2FqkZ6%2FNeymhjczyz9alIvJmMINFWwQ0ozvo2uNmUGWs8MuUVAX%2FNB4vdKXye1Nw%3D%3D"

class PowerBIKafkaConsumer:
    def __init__(self, topic, group_id, bootstrap_servers, power_bi_url):
        self.consumer = KafkaConsumer(
            topic,
            auto_offset_reset='latest',
            group_id=group_id,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.data = []
        self.power_bi_url = power_bi_url

    def consume_loop(self):
        while True:
            try:
                messages = self.consumer.poll(timeout_ms=1000)
                if messages:
                    for topic_partition, msgs in messages.items():
                        for message in msgs:
                            print(f"Received from Kafka topic '{topic_partition.topic}': {message.value}")
                            if not  message.value in self.data:
                                self.data.append(message.value)
                                response = requests.post(
                                    self.power_bi_url,
                                    json=self.data,
                                    headers={"Content-Type": "application/json"}
                                )
                                if response.status_code == 200:
                                    print("✅ Successfully sent to Power BI")
                                else:
                                    print(f"❌ Failed to send to Power BI: {response.status_code}, {response.text}")
                else:
                    print("No new messages, waiting...")
                    time.sleep(1)
            except Exception as e:
                print(f"Error consuming Kafka: {e}")
                time.sleep(2)

# Example usage
if __name__ == "__main__":
    power_bi_url = "https://api.powerbi.com/beta/..."  # Replace with actual full URL
    consumer = PowerBIKafkaConsumer(
        topic="cassandrastream",
        group_id="cassandrastream-group",
        bootstrap_servers='localhost:9092',
        power_bi_url=power_bi_url
    )
    consumer.consume_loop()
