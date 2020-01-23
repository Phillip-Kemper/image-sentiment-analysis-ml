import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {catchError, map} from "rxjs/operators";


@Injectable({
  providedIn: 'root'
})
export class ImageUploadService {
  constructor(private http:HttpClient) { }

  uploadImage(img,os) {

    console.log(os);
    if(os.equals('desktop')) {
      var url = "https://kemperino.com/api/upload"
    } else if(os.equals("ios")){
      var url = "https://kemperino.com/api2/upload"
    } else if(os.equals("android")){
      var url = "https://kemperino.com/api3/upload"
    }
    url = url.concat('/images/')
    return this.http.post(url, img)
      .pipe(map(res => res));

  }



  getImage(id){
    var url = "https://kemperino.com/api/upload"
    url = url.concat('/get/' + id);

    return this.http.get(url)
      .pipe(map(res => res));
  }
}
