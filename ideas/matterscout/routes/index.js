var express = require('express');
var router = express.Router();

var result_data = {
  1 : {
    "name": "Ice on camera",
    "description":"Ice on lens very early in the season.",
    "images": [1,2,3,4,5],
    "sensors": "2017-09-03_05:00:00",
    "time":"2017-09-03 05:00:00",
    "id":1,
    "severity":3.9
  },
  2 : {
    "name": "Climbers",
    "description":"Climbers moving away from the usual route.",
    "images": [1,2,3,4,5],
    "sensors": "2017-08-06_10:00:00",
    "time":"2017-08-06 10:00:00",
    "id":2,
    "severity":4.7
  },
  3 : {
    "name": "Climbers at night",
    "description":"Climbers moving at night and reaching very cose to the sensors.",
    "images": [1,2,3,4,5],
    "sensors": "2017-06-25_02:00:00",
    "time":"2017-06-25 02:00:00",
    "id":3,
    "severity":3.2
  }
};

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('home', {data: result_data});
});

router.get('/events', function(req, res, next) {
  res.render('index', {data: result_data});
});

router.get('/report', function(req, res, next) {
  res.render('report');
});

router.get('/event_detail/:id', function(req, res, next) {
  const event_id = req.params.id;
  res.render('event_detail', result_data[event_id]);
});

router.get('/event_summary', function(req, res, next) {
  const event_id = req.params.id;
  res.send(result_data);
});

router.post('/report', function(req, res, next) {
  var json = JSON.stringify(req.body)
  var fs = require('fs');
  fs.writeFile("json/" + req.body.title + ".json", json, 'utf8', () => {console.log("file received")});
  res.render('greetings');
});

module.exports = router;
