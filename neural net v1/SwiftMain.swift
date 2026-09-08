//
//  SwiftMain.swift
//  neural net v1
//
//  Created by Oliver Homer on 31/08/2026.
//

import Foundation
import SwiftUI
import AppKit
import NeuralApp

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct SwiftMain: App  {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    @StateObject private var engineBox = EngineBox (engine: AppEngine())

    var body: some Scene {
        WindowGroup {
            HStack(spacing: 0) {
                SwiftUIView(engineBox: engineBox)
                    .frame(width: 320)

                Divider()

                DrawingPad(engineBox: engineBox)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}
