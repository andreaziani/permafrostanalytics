var path = "/data/"+$("#get-event").val()+"/seismic_data.csv";


Plotly.d3.csv(path, function(err, rows){

    function unpack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }

    var trace1 = {
        type: "scatter",
        mode: "lines",
        name: 'Measurements EHE',
        x: unpack(rows, 'date'),
        y: unpack(rows, 'EHE'),
        line: {color: '#F44336'}
    };

    var trace2 = {
        type: "scatter",
        mode: "lines",
        name: 'Median EHE',
        x: unpack(rows, 'date'),

        y: unpack(rows, 'EHE-n'),
        line: {color: '#7F7F7F'}
    };

    /*
    var trace3 = {
        type: "scatter",
        mode: "lines",
        name: 'Measurements EHN',
        x: unpack(rows, 'date'),
        y: unpack(rows, 'EHN'),
        line: {color: '#66eef4'}
    };

    var trace4 = {
        type: "scatter",
        mode: "lines",
        name: 'Median EHN',
        x: unpack(rows, 'date'),

        y: unpack(rows, 'EHN-n'),
        line: {color: '#7f563b'}
    };


    var trace5 = {
        type: "scatter",
        mode: "lines",
        name: 'Measurements EHZ',
        x: unpack(rows, 'date'),
        y: unpack(rows, 'EHZ'),
        line: {color: '#f45eda'}
    };

    var trace6 = {
        type: "scatter",
        mode: "lines",
        name: 'Median EHZ',
        x: unpack(rows, 'date'),

        y: unpack(rows, 'EHZ-n'),
        line: {color: '#00587f'}
    };

*/

    var data = [trace1,trace2];

    var layout = {
        title: 'Seismic Data',
    };



    Plotly.newPlot('plot_data', data, layout, {showSendToCloud: true});
});