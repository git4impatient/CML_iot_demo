echo "calling the restAPI"
cat modelRESTapi.sh
curl -H "Content-Type: application/json" -X POST http://cdsw.52.0.206.26.nip.io/api/altus-ds-1/models/call-model -d '{"accessKey":"m9e2lxc5zh657ahph6hs9aoezgiigwzi","request":{"devid":"12345","sensor1":0.5,"sensor2":0.3,"sensor3":0.8,"sensor4":20}}'
echo ""
