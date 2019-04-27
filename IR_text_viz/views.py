from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from IR_text_viz.prep import *
import multiprocessing
def predictAndVisualize():
    # get the file name and save it somewhere
    # use the prev functions and predict
    # merge with raw text inp file
    pass

def upload(request):
    def getContext():
        folder = 'input/'
        p = preprocess('input')
        p.process_raw()
        p.getTestData()

        # predict
        testob = Classifier_LSTM('test_data.h5')
        df = testob.predict()
        grouped = df.groupby(['line'])
        data = []
        for l, group in grouped:
            line = {}
            for i, row in group.iterrows():
                line[i] = {
                    "class" : row['predict'],
                    "text" : row['text_x']
                }
            data.append(line)
        grouped = df.groupby('predict')

        classes = {}
        total = 0
        for key, group in grouped:
            total += len(group)
            classes[key] = len(group)

        print(total)

        for x in classes:
            classes[x] = classes[x] * 100 / total

        context =   {'data': data, 'classes':classes}
        filehandler = open('context.pkl', 'wb')
        pickle.dump(context, filehandler)
    # getContext()
    p = multiprocessing.Process(target=getContext)
    p.start()
    filehandler = open('context.pkl', 'rb')
    context = pickle.load(filehandler)
    # context = getContext()
    return render(request, "textViewer.html", context)



    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location=folder)
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)


        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'home.html')

