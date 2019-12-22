import { Component } from '@angular/core';
import {ImageUploadService} from './services/image-upload.service';
import { Observable, of } from 'rxjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

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

    this.imageUpload = this.imageService
      .uploadImage(this.selectedFile)
      .subscribe(res => console.log(res));
    this.secondScreen = true;

  }
}
