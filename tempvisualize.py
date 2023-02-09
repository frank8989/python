
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import numpy as np
import plotly
import plotly.graph_objs as go
import sys


"""
1. Remove invalid lines which include '-' as Max and Min temperature values.
2. Remove the useless year/hour/minitue part and values out of range.
3. Remove the Max/Min temperature values out of range.
"""


def filter_line(line):
    maxMin = []
    try:
        station, dateTime, tempMax, _, tempMin, *others = line.split()
    except ValueError as e:
        print("A problem: %s occurred when split the line: %s" % e % line)
        return []

    try:
        res = dateTime.split(":")
    except ValueError:
        print("A illegal formatted date occurred when split the line: %s" % dateTime)
        return []

    yearMonDay = res[0]
    if len(yearMonDay) != 8:
        return []
    if tempMax == "-" or tempMin == "-":
        return []
    tmpYearMonDay = int(yearMonDay)
    if tmpYearMonDay < 19500101 or tmpYearMonDay > 20501230:
        return []
    maxTemp = float(tempMax)
    minTemp = float(tempMin)
    if maxTemp < -50 or maxTemp > 50:
        return []
    if maxTemp < -50 or maxTemp > 50:
        return []
    maxMin = [station, yearMonDay, float(tempMax), float(tempMin)]
    return maxMin


"""
Keep stations'name to agent number mapping.
"""


def get_station_info(line):
    res = line.rsplit(" ", 7)
    station = {"Name": res[0], "Agent Number": res[1]}
    return station


"""
Read each line from the data source file and filter out invalid line:
1. Analyse head note to get stations information  
2. Skip footer note using isdigit to filter
3. Create a thread pool to handle each station in parallel.
"""


def ingest_file(fileName, executor, predictTasks):
    tasks = []
    aucklandThisYearData = []
    lines = []
    stations = []
    stationNoteFound = False
    dataComplete = False
    lastStation = ""
    with open(fileName, "r") as inputFile:
        for line in inputFile:
            station = ""
            if line[0].isdigit():
                lineData = filter_line(line)
                if lineData:
                    station = lineData[0]
                    if lastStation != "" and station != lastStation:
                        task = executor.submit(meanMaxMin, lines.copy())
                        tasks.append(task)
                        lines.clear()
                    lines.append(lineData)
                    if station == "41351" and int(lineData[1]) >= 20220101:
                        aucklandThisYearData.append(lineData)
            elif "Name" in line:
                stationNoteFound = True
            elif "Note" in line:
                dataComplete = True
            elif stationNoteFound and not dataComplete:
                stations.append(get_station_info(line))
            else:
                continue
            lastStation = station
    task = executor.submit(meanMaxMin, lines)
    tasks.append(task)
    task = executor.submit(calculate_model, aucklandThisYearData)
    predictTasks.append(task)
    return stations, tasks


"""
Remove values outside of 5 standard deviations and calculate the mean values.
"""


def calcu_mean(maxTem, minTem):
    meanMax = np.mean(maxTem)
    meanMin = np.mean(minTem)
    stdMax = np.std(maxTem, ddof=1)
    stdMin = np.std(minTem, ddof=1)
    highMax = meanMax + 5 * stdMax
    lowMax = meanMax - 5 * stdMax
    highMin = meanMin + 5 * stdMin
    lowMin = meanMin - 5 * stdMin
    maxTem = [x for x in maxTem if x <= highMax and x >= lowMax]
    minTem = [x for x in minTem if x <= highMin and x >= lowMin]
    meanMax = np.mean(maxTem)
    meanMin = np.mean(minTem)
    return meanMax, meanMin


"""
Scan the whole data list and calulate mean values for each month
Preconditon: the data records are sorted firstly by station agent number and then by data generated time
"""


def meanMaxMin(dataList):
    lastYearMon = ""
    meanMaxMinTemList = []
    maxTem = []
    minTem = []
    length = len(dataList)
    for i, data in enumerate(dataList):
        station = data[0]
        yearMonDay = data[1]
        yearMon = yearMonDay[:6]

        # First month or the same month
        if yearMon == lastYearMon or lastYearMon == "":
            maxTem.append(data[2])
            minTem.append(data[3])

        # Start a new month or the last record
        if (i != 0 and yearMon != lastYearMon) or (i + 1) == length:
            meanMaxTem, meanMinTem = calcu_mean(maxTem, minTem)
            if i + 1 == length:
                lastYearMon = yearMon
            meanMaxMin = [data[0], lastYearMon, meanMaxTem, meanMinTem]
            meanMaxMinTemList.append(meanMaxMin)
            maxTem.clear()
            minTem.clear()
        lastYearMon = yearMon
    return meanMaxMinTemList


def generate_trace(fileName, executor, predictTasks):
    data = []
    stations, tasks = ingest_file(fileName, executor, predictTasks)
    mean = []
    try:
        for task in as_completed(tasks):
            mean.extend(task.result())
    except Exception:
        print("Wait for tasks to complete with an exception.")
        return data

    months = []
    maxTem = []
    minTem = []
    lastStation = ""
    length = len(mean)
    stationName = ""
    for i, meanData in enumerate(mean):
        station = meanData[0]
        if station == lastStation or lastStation == "":
            months.append(meanData[1])
            maxTem.append(meanData[2])
            minTem.append(meanData[3])
        if (i != 0 and station != lastStation) or (i + 1) == length:
            if (i + 1) == length:
                lastStation = station

            for tmpStation in stations:
                if tmpStation["Agent Number"] == lastStation:
                    stationName = tmpStation["Name"]
                    break

            trace = go.Scatter(
                x=months,
                y=maxTem,
                mode="lines",
                name=stationName + " max",
                marker=dict(
                    size=10,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
            trace = go.Scatter(
                x=months,
                y=minTem,
                mode="lines",
                name=stationName + " min",
                marker=dict(
                    size=10,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
            months.clear()
            maxTem.clear()
            minTem.clear()
        lastStation = station
    return data


"""
Use the whole year's daily data to train the polynomial model and predict the future week's temperature  
"""


def calculate_model(aucklandThisYearData):
    date = []
    maxTem = []
    minTem = []
    degree = 4
    predictMaxTem = []
    predictMinTem = []
    originalX = []

    if not aucklandThisYearData:
        return ()

    for data in aucklandThisYearData:
        date.append(int(data[1]))
        maxTem.append(data[2])
        minTem.append(data[3])
    length = len(date)
    lastDay = date[length - 1]
    X = [i for i in range(len(date))]
    maxCoef = np.polyfit(X, maxTem, degree)
    minCoef = np.polyfit(X, minTem, degree)
    originalX = X

    """
    Predict more 14 days in the future
    """

    for i in range(14):
        X.append(length + i)
        date.append(lastDay + i + 1)

    for i in range(len(X)):
        value = maxCoef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * maxCoef[d]
        predictMaxTem.append(value)
    for i in range(len(X)):
        value = minCoef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * minCoef[d]
        predictMinTem.append(value)

    return X, predictMaxTem, predictMinTem, originalX, maxTem, minTem


def predict(executor, predictTasks):
    data = []
    try:
        for task in as_completed(predictTasks):
            daySeq, predictMaxTem, predictMinTem, days, maxTem, minTem = task.result()
            if not daySeq:
                continue
            trace = go.Scatter(
                x=daySeq,
                y=predictMinTem,
                mode="lines",
                name="Auckland Predition Min",
                marker=dict(
                    size=2,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
            trace = go.Scatter(
                x=daySeq,
                y=predictMaxTem,
                mode="lines",
                name="Auckland Predition Max",
                marker=dict(
                    size=2,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
            trace = go.Scatter(
                x=days,
                y=minTem,
                mode="lines",
                name="Auckland Daily Min",
                marker=dict(
                    size=2,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
            trace = go.Scatter(
                x=days,
                y=maxTem,
                mode="lines",
                name="Auckland Daily Max",
                marker=dict(
                    size=2,
                    color=np.random.randn(500),
                    line=dict(width=2),
                    colorscale="Viridis",
                    showscale=True,
                ),
            )
            data.append(trace)
        executor.shutdown()
    except Exception:
        print("Wait for predict tasks to complete with an exception.")
        return data

    return data


"""
    Call the python programe liked this :  watherdriver.py datafile.
"""


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Call the python programe like this: watcheriver.py datafile")
        return
    dataFile = sys.argv[1]
    executor = ThreadPoolExecutor(10)
    predictTasks = []
    data = generate_trace(dataFile, executor, predictTasks)
    if data:
        layout = go.Layout(
            title=dict(
                text="Four Stations Monthly Average Max and Min Temperature",
                x=0.5,
                xanchor="center",
                xref="paper",
            ),
            font={"size": 22, "family": "sans-serif", "color": "black"},
            titlefont={"size": 20, "color": "blueviolet"},
            xaxis={
                "title": "Date",
                "titlefont": {"size": 30},
                "showgrid": True,
                "gridcolor": "tomato",
            },
            yaxis={
                "title": "Temperature(C)",
                "titlefont": {"size": 20},
                "showgrid": True,
                "gridcolor": "mediumorchid",
                "showline": False,
                "zeroline": True,
            },
            legend={"x": 0.9, "y": 1},
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename="5-years-4-stations.html")

    data = predict(executor, predictTasks)
    if data:
        layout = go.Layout(
            title=dict(
                text="Auckland Daily Max and Min Temperature in 2022",
                x=0.5,
                xanchor="center",
                xref="paper",
            ),
            font={"size": 22, "family": "sans-serif", "color": "black"},
            titlefont={"size": 20, "color": "blueviolet"},
            xaxis={
                "title": "Date",
                "titlefont": {"size": 30},
                "showgrid": True,
                "gridcolor": "tomato",
            },
            yaxis={
                "title": "Temperature(C)",
                "titlefont": {"size": 20},
                "showgrid": True,
                "gridcolor": "mediumorchid",
                "showline": False,
                "zeroline": True,
            },
            legend={"x": 0.9, "y": 1},
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename="predict-nextweek.html")


if __name__ == "__main__":
    main()