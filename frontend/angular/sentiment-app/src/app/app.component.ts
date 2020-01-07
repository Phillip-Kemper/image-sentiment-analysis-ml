import {Component, ElementRef, ViewChild} from '@angular/core';
import {ImageUploadService} from './services/image-upload.service';
import { Observable, of } from 'rxjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  private previewUrl: string | ArrayBuffer;

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
      this.previewUrl = reader.result;
    }




    this.imageUpload = this.imageService
      .uploadImage(this.selectedFile)
      .subscribe(res => console.log(res));
    this.secondScreen = true;

  }
}
