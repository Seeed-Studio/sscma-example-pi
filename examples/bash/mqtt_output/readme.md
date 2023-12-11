# MQTT sink

## Dependencies
1. mosquitto
2. libpaho-mqtt-dev

## steps
1. run mosquitto
```bash
mosquitto -v
```

2. run the mqtt sink
```bash
./run.sh
```

3. see the result in mqtt client
  you can use any mqtt client to subscribe the topic "sscma_yolov5", and you will see the result in json format.