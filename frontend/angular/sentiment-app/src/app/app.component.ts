import {Component, ElementRef, ViewChild} from '@angular/core';
import {ImageUploadService} from './services/image-upload.service';
import {Observable, of} from 'rxjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  private previewUrl: any;
  private loadedImg;

  constructor(private imageService: ImageUploadService) {

  }

  title = 'sentiment-app';
  secondScreen = false;
  selectedFile = null;
  imageUpload = null;


  onFileSelected(event) {
    this.selectedFile = event.target.files[0];
    if (this.selectedFile != null) {

    }

  }

  onUpload() {
    console.log(this.selectedFile); // You can use FormData upload to backend server

    let image = this.selectedFile

    var reader = new FileReader();

    reader.readAsDataURL(this.selectedFile);
    reader.onload = (_event) => {
      this.loadedImg = reader.result;
      console.log(this.loadedImg);
    }
    console.log(image);


     this.imageUpload = this.imageService
       .uploadImage(this.selectedFile)
       .subscribe(res => console.log(res));
     this.secondScreen = true;

  }

  grayscale(image, bPlaceImage)
{
  image = image.loadedImg
  image = document.getElementById("myimg")
  console.log("now start")
  console.log(image)
  var myCanvas=document.createElement("canvas");
  var myCanvasContext=myCanvas.getContext("2d");

  var imgWidth=image.width;
  var imgHeight=image.height;
  // You'll get some string error if you fail to specify the dimensions
  myCanvas.width= imgWidth;
  myCanvas.height=imgHeight;
  //  alert(imgWidth);
  myCanvasContext.drawImage(image,0,0,48,48);

  // This function cannot be called if the image is not rom the same domain.
  // You'll get security error if you do.
  var imgPixels=myCanvasContext.getImageData(0,0, imgWidth, imgHeight);

  // This loop gets every pixels on the image and

              for(var y = 0; y < imgHeight; y++){
                for(var x = 0; x < imgWidth; x++){
                    var i = (y * 4) * imgWidth + x * 4;
                    var avg = (imgPixels.data[i] + imgPixels.data[i + 1] + imgPixels.data[i + 2]) / 3;
                    imgPixels.data[i] = avg;
                    imgPixels.data[i + 1] = avg;
                    imgPixels.data[i + 2] = avg;
                }
            }

  console.log("go on ")

            myCanvasContext.putImageData(imgPixels, 0, 0, 0, 0, imgPixels.width, imgPixels.height);

  //  for (var j=0; j<imageData.height; j++)
  //  {
  //    for (var i=0; i<imageData.width; i++)
  //    {
  //       var index=(i*4)*imageData.width+(j*4);
  //       var red=imageData.data[index];
  //       var green=imageData.data[index+1];
  //       var blue=imageData.data[index+2];
  //       var alpha=imageData.data[index+3];
  //       var average=(red+green+blue)/3;
  // 	    imageData.data[index]=average;
  //       imageData.data[index+1]=average;
  //       imageData.data[index+2]=average;
  //       imageData.data[index+3]=alpha;
  //     }
  //   }

    if (bPlaceImage)
	{
	  var myDiv=document.createElement("div");
	     myDiv.appendChild(myCanvas);
	  image.parentNode.appendChild(myCanvas);
	}
    console.log("fin")
  console.log(myCanvas.toDataURL())
  return myCanvas.toDataURL();
  }

  gray(input,some) {
      var myimage = input
      myimage = document.getElementById("myimg")

            var cnv = document.createElement('canvas');
        var cnx = cnv.getContext('2d');

          var width=myimage.width;
  var height=myimage.height;
  // You'll get some string error if you fail to specify the dimensions
  cnv.width= width;
  cnv.height=height;

            cnx.drawImage(myimage, 0 , 0);
            var width = input.width;
            var height = input.height;
            var imgPixels = cnx.getImageData(0, 0, width, height);



        }



}
