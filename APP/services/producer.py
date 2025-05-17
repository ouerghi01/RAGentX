import json
import time
from kafka import  KafkaProducer

# Initialize Cassandra
# # Set retention to 1 second
#kafka-configs.sh --bootstrap-server localhost:9092 --entity-type topics --entity-name your-topic-name --alter --add-config retention.ms=1000


class Producer:
    def __init__(self, topic ):
        self.topic = topic
        self.stack = []
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_message(self, message):
        future = self.producer.send(self.topic, value=message)
        try:
            record_metadata = future.get(timeout=10)
            print(f"✅ Successfully sent to Kafka topic '{record_metadata.topic}', partition {record_metadata.partition}, offset {record_metadata.offset}")
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
    def poll_cassandra(self,
    timestamp,partition_id,answer,question,the_time_question_sended,user
    ):
        try:
            data = {
                    "time_question_answered": str(timestamp),
                    "partition_id": str(partition_id),
                    "response": answer,
                    "question": question,
                    "username": user,
                    "time_question_sended": str(the_time_question_sended),
                    "source": "cassandra"
                    }
            if not data in self.stack:
                # Check if the data is already in the list to avoid duplicates
                self.send_message(data)
                self.stack.append(data)
        except Exception as e:
                print(f"Error polling Cassandra: {e}")
                time.sleep(5)

