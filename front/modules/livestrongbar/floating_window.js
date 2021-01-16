// Project https://github.com/riversun/JSFrame.js
// source: https://riversun.github.io/JSFrame.js/public/
// (*)window_control.html - Window size control manually * This example code is for v1.5 or later and will be updated in the future

const jsFrame = new JSFrame({
    horizontalAlign: 'left',
    verticalAlign: 'top',
});

function createStrongLiveBar() {

    const frame = jsFrame.create({
        name: `Win`,
        title: `Strong Buy / Sell`,
        left: 20, top: 40, width: 820, height: 570, minWidth: 200, minHeight: 110,
        movable: true,//Enable to be moved by mouse
        resizable: false,//Enable to be resized by mouse
        appearanceName: 'material',
        appearanceParam: {
            border: {
                shadow: '2px 2px 10px  rgba(0, 0, 0, 0.5)',
                width: 0,
                radius: 6,
            },
            titleBar: {
                color: 'white',
                background: '#d3d3d3',
                height: 30,
                fontSize: 14,
                buttonWidth: 36,
                buttonHeight: 16,
                buttonColor: 'white',
            }
        },
        style: {
            //backgroundColor: 'rgba(255,255,255,0.8)',
            overflow: 'auto'
        },
        url: 'iframe/livestrongbar.html',
        //html: document.getElementById('livebar0').innerHTML
    }).show();


    frame.setControl({
        maximizeButton: 'maximizeButton',
        demaximizeButton: 'restoreButton',
        minimizeButton: 'minimizeButton',
        deminimizeButton: 'deminimizeButton',
        animation: true,
        animationDuration: 200,

    });


    frame.on('maximizeButton', 'click', (_frame, evt) => {
        _frame.control.doMaximize({
            hideTitleBar: false,
            duration: 200,
            restoreKey: 'Escape',
            restoreDuration: 100,
            callback: (frame, info) => {
                frame.requestFocus();
            },
            restoreCallback: (frame, info) => {
                // jsFrame.showToast({
                //     text: frame.getName() + ' ' + info.eventType
                // });
            },
        });
    });

    frame.on('restoreButton', 'click', (_frame, evt) => {
        _frame.control.doDemaximize(
            {
                duration: 200,
                callback: (frame, info) => {
                    // jsFrame.showToast({
                    //     text: frame.getName() + ' ' + info.eventType
                    // });
                }
            });
    });

    frame.on('minimizeButton', 'click', (_frame, evt) => {

        _frame.control.doMinimize({
            duration: 200,
            callback: (frame, info) => {
                // jsFrame.showToast({
                //     text: frame.getName() + ' ' + info.eventType
                // });
            }
        });

    });

    frame.on('deminimizeButton', 'click', (_frame, evt) => {
        _frame.control.doDeminimize({
            duration: 200,
            callback: (frame, info) => {
                // jsFrame.showToast({
                //     text: frame.getName() + ' ' + info.eventType
                // });
            }
        });
    });

    frame.on('closeButton', 'click', (_frame, evt) => {
        _frame.control.doHide({
            duration: 100,
            align: 'CENTER_BOTTOM',
            callback: (frame, info) => {
                // jsFrame.showToast({
                //     text: frame.getName() + ' ' + info.eventType
                // });
                _frame.closeFrame();
            }
        });
    });
}


function restoreStrongLiveBar() {
    var frame = jsFrame.getWindowByName('Win');
    console.log(frame)
    frame.control.doDehide();
}


createStrongLiveBar();
