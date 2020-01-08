import {Component, ElementRef, ViewChild} from '@angular/core';
import {ImageUploadService} from './services/image-upload.service';
import { Observable, of } from 'rxjs';



@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  private previewUrl: any;
  private loadedImg;

  constructor(private imageService: ImageUploadService){

  }

  title = 'sentiment-app';
  secondScreen = false;
  selectedFile = null;
  imageUpload = null;


  onFileSelected(event)
  {
    this.selectedFile = event.target.files[0];
    if(this.selectedFile!=null){

    }

  }

  onUpload()
  {
    console.log(this.selectedFile); // You can use FormData upload to backend server

    let image = this.selectedFile

    const size = image.size;
    let width;
    let height;

    var reader = new FileReader();

    reader.readAsDataURL(this.selectedFile);
    reader.onload = (_event) => {
      this.loadedImg = reader.result;
    }
    console.log(image);

    //let myImageUrl = URL.createObjectURL(this.selectedFile);
    //let myImage = new Image();
    //myImage.src = myImageUrl;
    //myImage.onload = (_even) => {
    //  this.previewUrl = this.grayscale(image, true);
    //}
    this.wait(2500);



    const x = document.getElementById('myimg');

    var image1 = new Image();

    console.log(x)

    this.previewUrl = this.grayscale(image1,true);
    console.log(this.previewUrl)


    this.imageUpload = this.imageService
      .uploadImage(this.selectedFile)
      .subscribe(res => console.log(res));
    this.secondScreen = true;

  }

  grayscale(image, bPlaceImage)
{

  console.log(image)
  var myCanvas=document.createElement("canvas");
  var myCanvasContext=myCanvas.getContext("2d");

  var imgWidth=image.width;
  var imgHeight=image.height;
  // You'll get some string error if you fail to specify the dimensions
  myCanvas.width= imgWidth;
  myCanvas.height=imgHeight;
  //  alert(imgWidth);
  myCanvasContext.drawImage(image,0,0);

  // This function cannot be called if the image is not rom the same domain.
  // You'll get security error if you do.
  var imageData=myCanvasContext.getImageData(0,0, imgWidth, imgHeight);

  // This loop gets every pixels on the image and
    for (let j=0; j<imageData.height; j++)
    {
      for (let i=0; i<imageData.width; i++)
      {
         var index=(i*4)*imageData.width+(j*4);
         var red=imageData.data[index];
         var green=imageData.data[index+1];
         var blue=imageData.data[index+2];
         var alpha=imageData.data[index+3];
         var average=(red+green+blue)/3;
   	    imageData.data[index]=average;
         imageData.data[index+1]=average;
         imageData.data[index+2]=average;
         imageData.data[index+3]=alpha;
       }
     }

    if (bPlaceImage)
	{
	  var myDiv=document.createElement("div");
	     myDiv.appendChild(myCanvas);
	  image.parentNode.appendChild(myCanvas);
	}
	return myCanvas.toDataURL();
  }


  wait(ms){
   var start = new Date().getTime();
   var end = start;
   while(end < start + ms) {
     end = new Date().getTime();
  }
}
}
