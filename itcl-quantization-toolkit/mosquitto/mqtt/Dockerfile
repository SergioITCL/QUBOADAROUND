

FROM eclipse-mosquitto:latest
COPY ./cfg/mosquitto.conf /mosquitto/config/mosquitto.conf

RUN touch  /mosquitto/config/mosquitto.pwd

ARG MQTT_PWD
ARG MQTT_USER
RUN mosquitto_passwd -b /mosquitto/config/mosquitto.pwd ${MQTT_USER} ${MQTT_PWD}

EXPOSE 1883
EXPOSE 9001
